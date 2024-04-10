"""Pretrain ppo utilities."""

from datetime import datetime
import bisect
import math
import sys
import time
import json
import copy
import random
import numpy as np
# The earliest we can measure the start time.
_TRAIN_START_TIME = time.time()

import torch
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP
from deepspeed.runtime.pipe.engine import PipelineEngine
from deepspeed.runtime.pipe.engine import schedule
from torch.nn import functional as F
from torch.utils.data import DataLoader
from megatron import prompt_template,remove_prompt_template

from megatron import get_args
from megatron import get_timers
from megatron import get_tensorboard_writer
# from megatron import get_current_global_batch_size
from megatron import get_num_microbatches
# from megatron import is_last_rank
from megatron import update_num_microbatches
from megatron import mpu
from tqdm import tqdm
from megatron import get_tokenizer
from megatron import print_rank_0,print_with_rank
# from megatron import print_rank_last
from megatron.checkpointing import load_checkpoint
from megatron.checkpointing import save_checkpoint
from megatron.model.module import Float16Module
# from megatron.optimizer import get_megatron_optimizer
from megatron.initialize import initialize_megatron
# from megatron.initialize import write_args_to_tensorboard, log_restart_to_tensorboard
from megatron.learning_rates import AnnealingLR
from megatron.model.distributed import DistributedDataParallel as LocalDDP
# from megatron.utils import check_adlr_autoresume_termination, get_parameters_in_billions
from megatron.utils import unwrap_model, found_kill_switch
# from megatron.data.data_samplers import build_pretraining_data_loader
# from megatron.utils import calc_params_l2_norm
from megatron.schedules import forward_backward_no_pipelining
from megatron.schedules import forward_backward_pipelining_without_interleaving
from megatron.schedules import forward_backward_pipelining_with_interleaving
# from megatron.utils import report_memory, flops_calculator
# from megatron.global_vars import codecarbon_tracker_start, codecarbon_tracker_stop
# from megatron.data.dataset_utils import analyze_data_prefix
from megatron.text_generation_ppo import generate_samples as generate_samples_ppo
from megatron.data.ppo_dataset import RolloutDataset
from megatron.model.gpt_model_ppo  import GPTModelWithPPOValueHead
from megatron.training import setup_model_and_optimizer,print_datetime,save_checkpoint_and_time
import torch.distributed as dist
import deepspeed
# from torch.nn.utils.rnn import pad_sequence

WANDB_PADDING = -1


def is_last_stage_and_scr_rank():
  """判断当前rank是否属于最后一个stage,并且在张量并行中排第一
  """
  return  mpu.is_pipeline_last_stage() and \
      dist.get_rank()==mpu.get_tensor_model_parallel_src_rank()

def is_first_stage_and_scr_rank():
  """判断当前rank是否属于第一个stage,并且在张量并行中排第一
  """
  return mpu.is_pipeline_first_stage() and \
      dist.get_rank()==mpu.get_tensor_model_parallel_src_rank()

def is_tensor_model_parallel_src_rank():
  return dist.get_rank()==mpu.get_tensor_model_parallel_src_rank()

def broadcast_prompts_across_pp_tp_group(prompts:list):
  """
  沿着模型并行维度进行list[str]变量的广播
  Args:
      prompt_dataloader (torch.utils.data.DataLoader): _description_
  """
  assert isinstance(prompts,list) and isinstance(prompts[0],str), prompts
  dist.broadcast_object_list(
      prompts,
      src=mpu.get_pipeline_model_parallel_last_rank(),
      group=mpu.get_pipeline_model_parallel_group()
    )

  dist.broadcast_object_list(
      prompts,
      src=mpu.get_tensor_model_parallel_src_rank(),
      group=mpu.get_tensor_model_parallel_group()
    )
  
  return prompts

def broadcast_list_tensor_across_pp_src_group(input_tensors_list:list):
  pmp_last_rank = mpu.get_pipeline_model_parallel_last_rank()
  if is_last_stage_and_scr_rank():
    assert dist.get_rank() == pmp_last_rank
    dist.broadcast_object_list(
            [input_tensors.cpu() for input_tensors in input_tensors_list],
            src=pmp_last_rank,
            group=mpu.get_pipeline_model_parallel_group()
    )
  elif mpu.get_tensor_model_parallel_src_rank() == dist.get_rank():
  
    dist.broadcast_object_list(
            input_tensors_list,
            src=pmp_last_rank,
            group=mpu.get_pipeline_model_parallel_group()
    )
    input_tensors_list = [input_tensors.cuda() for input_tensors in input_tensors_list]
  return input_tensors_list


# def stack_dicts(stats_dicts):
#   """Stack the values of a dict."""
#   results = dict()
#   for k in stats_dicts[0]:
#       stats_list = [torch.flatten(d[k]) for d in stats_dicts]
#       results[k] = pad_sequence(stats_list, batch_first=True, padding_value=WANDB_PADDING)
#   return results

def record_generated_texts(prompts,generated_texts,rewards,iteration):
  writer = get_tensorboard_writer()
  markdown_table = """
  | prompt | response | score |
  | :----: | :----: | :----: |"""
  for prompt, generated_text,reward in zip(prompts,generated_texts,rewards):
    prompt = prompt.replace("\n","<br />")
    generated_text = generated_text.replace("\n","<br />")
    row = '\n' + f'| {prompt} | {generated_text} | {round(reward,3)}|'
    markdown_table += row 
  writer.add_text(f'ppo/text_generate_sample_step_{iteration}',markdown_table,iteration)


def record_step_stats(all_stats:list,iteration,extra_kwargs=None):
  def average(scalars:list):
    return sum(scalars)/len(scalars)
  stats = {}
  for k in all_stats[0].keys():
    stats[k] = average([l[k] for l in all_stats])
  
  # log to tensorboard
  writer = get_tensorboard_writer()
  if writer is not None:
    for k,v in stats.items():
      writer.add_scalar(k, v,iteration)
    
    if extra_kwargs is not None:
      assert isinstance(extra_kwargs,dict)
      for k,v in extra_kwargs.items():
        writer.add_scalar(k, v,iteration)

  
def ppo_train(prompts_dataset_provider,
              model_provider,
              forward_step_func,
              reward_fn,
              ref_model_fn,
              kl_ctl_provider,
              extra_args_provider=None,
              param_groups_provider=None,
              learning_rate_scheduler_provider=None,
              args_defaults={}):
  """
      train program for ppo algorithm.
      This function will run the followings in the order provided:
          1) initialize Megatron.
          2) setup model, optimizer and lr schedule using the model_provider.
          3) call train_val_test_data_provider to get train/val/test datasets.
          4) train the model.
      prompts_iter_provider: a function that provide train/val/test prompts
      model_provider: a function that provide actor_critic_model,
      reward_fn: a method to calculate reward of the sampling.
      ref_model_fn: 使用引用模型进行计算
      extra_args_provider: a function that takes a parser and adds arguments
              to it. It is used for programs to add their own arguments.
          args_defaults: a dictionary from argument-name to argument-value. It
              to set already parse arguments.
      """
  
  # 初始化megatron
  initialize_megatron(extra_args_provider=extra_args_provider,
                      args_defaults=args_defaults)

  args = get_args()
  tokenizer = get_tokenizer()
  assert not args.deepspeed, 'current version not support deepspeed'
  # assert args.micro_batch_size == 1

  assert mpu.get_data_parallel_world_size() == 1, 'currently, no support for ddp'

  global _TRAIN_START_TIME
  start_time_tensor = torch.cuda.FloatTensor([_TRAIN_START_TIME])
  torch.distributed.all_reduce(start_time_tensor,
                               op=torch.distributed.ReduceOp.MIN)
  _TRAIN_START_TIME = start_time_tensor.item()
  print_rank_0('time to initialize megatron (seconds): {:.3f}'.format(
      time.time() - _TRAIN_START_TIME))

  # 实例化model，optimizer, lr_scheduler
  model, optimizer, lr_scheduler = setup_model_and_optimizer(
    model_provider,
    param_groups_provider=param_groups_provider,
    learning_rate_scheduler_provider=learning_rate_scheduler_provider
    )
  
  
  # for param_group in optimizer.param_groups:
  #   print('param group lr after initial',param_group['lr'], len(param_group['params']))
  
  # print(sg)
  
  lr_scheduler = None  # 暂时不进行学习率下降策略
  
  # print(type(model),len(model))
  print_datetime('after megatron is initialized')
  timers = get_timers()
 
  iteration = 0

  unwrapped_model = unwrap_model(model[0], (torchDDP, LocalDDP, Float16Module))
  ppo_config = unwrapped_model.ppo_config
  num_rollouts = ppo_config['num_rollouts']

  # args.recompute = True
  args.temperature = ppo_config['gen_kwargs']['temperature']
  # max_new_tokens = ppo_config['gen_kwargs']['max_new_tokens']
  args.top_k =  ppo_config['gen_kwargs']['top_k']
  args.top_p =  ppo_config['gen_kwargs']['top_p']
  args.greedy = False
  args.recompute = False
  args.ppo_iteration = 0
  args.ppo_learn_iteration = 0
  args.ppo_config = ppo_config


  prompt_dataset = prompts_dataset_provider()
  prompt_dataloader = torch.utils.data.DataLoader(
    prompt_dataset,
    batch_size=num_rollouts, shuffle=True,
  )
  # 用来调节kl散度的惩罚力度
  kl_ctl = kl_ctl_provider(ppo_config)

  tokenizer = get_tokenizer()

  # logprobs_of_labels = GPTModelWithPPOValueHead.logprobs_of_labels
#  
  # print_with_rank('start training')
  while iteration < args.train_iters:
    # 特殊情况下关闭训练
    if found_kill_switch():
        save_checkpoint_and_time(iteration, model, optimizer, lr_scheduler)
        print_datetime(f"Detected kill switch at {args.kill_switch_path}. Exiting")
        sys.exit()
    # 对最后一个stage的第一个张量rank进行采样
    # 之后需要将这个结果送到第一个stage中
    # print('正在和环境互动中')
    # 利用\theta_old 跟环境互动得到采样

    for prompts in prompt_dataloader:
      if iteration >= args.train_iters:
        print_rank_0('train finish')
        save_checkpoint(iteration, model, optimizer, lr_scheduler)
        return 
      # 最后一个batch的不需要
      if len(prompts) != num_rollouts:
        break
      ppo_learn(
        prompts,
        tokenizer,
        args.seq_length,
        args.micro_batch_size,
        model,
        unwrapped_model,
        forward_step_func,
        reward_fn,
        ref_model_fn,
        optimizer,
        lr_scheduler,
        timers,
        ppo_config['gen_kwargs'],
        ppo_config,
        kl_ctl
      )
      # 保存
      if iteration !=0 and iteration % args.save_interval ==0:
        print_rank_0(f'保存模型中: {iteration}')
        save_checkpoint(iteration, model, optimizer, lr_scheduler)
    
      iteration += 1
  
def ppo_learn(
  prompts,
  tokenizer,
  seq_len,
  micro_batch_size,
  model,
  unwrapped_model,
  forward_step_func,
  reward_fn,
  ref_model_fn,
  optimizer,
  lr_scheduler, 
  timers,
  generate_kwargs:dict,
  ppo_config,
  kl_ctl
  ):
  """_summary_

  Args:
      prompts (_type_): _description_
      tokenizer (_type_): _description_
      seq_len (_type_): _description_
      micro_batch_size (_type_): _description_
      model (_type_): _description_
      unwrapped_model (_type_): _description_
      forward_step_func (_type_): _description_
      reward_fn (_type_): _description_
      ref_model_fn (_type_): _description_
      optimizer (_type_): _description_
      lr_scheduler (_type_): _description_
      timers (_type_): _description_
      generate_kwargs (dict): _description_
      ppo_config (_type_): _description_
  """
  
  args = get_args()
  max_new_tokens = generate_kwargs['max_new_tokens']
  logprobs_of_labels = GPTModelWithPPOValueHead.logprobs_of_labels

  query_tensors_list = []
  cur_prompt_list = prompts
  response_tensors_list = []
  old_logprob_list = []
  old_values_list = []
  old_generated_text_list = []
  old_rewards_list = []
  response_tensor_len_list = []
  input_tensors_list = []

  # 同步prompts
  prompts = broadcast_prompts_across_pp_tp_group(prompts)
  
  # print_with_rank(prompts)
  # 生成的时候需要临时关闭一下checkpoint_activations
  checkpoint_activations = args.checkpoint_activations
  args.checkpoint_activations = False
  t0 = time.time()
  ppo_generate_batch_size = getattr(args,'ppo_generate_batch_size',micro_batch_size)
  ret = generate_batched(
    prompt_template(prompts) if not getattr(args,'generate_without_template',False) else prompts,
    unwrapped_model, 
    max_new_tokens,
    tokenizer,
    tokenizer.eod,
    seq_len,
    batch_size=ppo_generate_batch_size,
    recompute=args.recompute,
    greedy=args.greedy,
    temperature=args.temperature,
    top_k=args.top_k,
    top_p=args.top_p,
    )

  print_rank_0(f'sample {len(prompts)},max new tokens: {max_new_tokens} time: {time.time()-t0}')
  
  # print_with_rank(ret)
  args.checkpoint_activations = checkpoint_activations
  
  # 最后一个stage的才会进行计算
  if ret is not None: # and ret['lm_logits'] is not None:
    query_tensors_list = ret['query_tensors']
    response_tensors_list = ret['response_tensors']
    old_logprob_list = [
      logprobs_of_labels(
        lm_logit,
        response_tensor
      ) 
      for lm_logit, response_tensor in zip(ret['lm_logits'],response_tensors_list)
      ]
    old_values_list = ret['values']
    old_generated_text_list = ret['generated_texts']
    response_tensor_len_list = [response_tensor.shape[1] for response_tensor in response_tensors_list]
    for i in range(len(response_tensor_len_list)):
      # print('response len',response_tensor_len_list[i],'old_values',old_values_list[i].shape[1])
      if response_tensors_list[i].numel() == 0:
        print('empty response tensor, give up current learn !!!!',prompts[i],query_tensors_list[i],response_tensors_list[i])
      
  # 计算reward与score
  kl_mean = None
  if is_last_stage_and_scr_rank():
    texts = [prompt + generated_text for prompt,generated_text in zip(cur_prompt_list,old_generated_text_list)]
    query_py_list = [q.tolist()[0] for q in query_tensors_list]
    response_py_list = [r.tolist()[0] for r in response_tensors_list]
    # 计算引用模型的logits
    # print('正在计算ref_logits_list',query_py_list,response_py_list)
    ref_logprob_list = ref_model_fn({
      'query_tensors':query_py_list,'response_tensors':response_py_list
      })
    
    # print('正在计算rreward_fn')
    # print(texts)
    reward_list= reward_fn(texts)
    old_rewards_list = copy.deepcopy(reward_list)
    
    # 对reward进行归一化
    old_rewards_list = np.array(old_rewards_list)
    old_rewards_list = (old_rewards_list-old_rewards_list.mean())/old_rewards_list.std()
    old_rewards_list = old_rewards_list.tolist()
    
    
    for i,prompt in enumerate(prompts[:5]):
      print_with_rank(f'正在采样: {i},{prompt}->{ret["generated_texts"][i]},score: {old_rewards_list[i]}')
    
     
    max_text_len = max([len(text) for text in texts])

    # print(f'text maxlen:',max_text_len)
    # for i in range(len(texts)):
    #   if len(texts[i]) == max_text_len: 
    #     print(
    #       f'最大text: {prompts[i]}->{ret["generated_texts"][i]},score: {old_rewards_list[i]}',
    #       'response tensor shap',
    #       ret['response_tensors'][i],
    #     )
    #     break
  
    # 将kl散度作为惩罚项放到reward中
    old_rewards_list,non_score_rewards_list,kls = compute_reward_with_kl_divergence(
        old_rewards_list,
        ref_logprob_list,
        old_logprob_list,
        kl_ctl
      )
  

    # print('response_tensors max len:', max([response_tensor.shape[1] for response_tensor in ret['response_tensors']]))
    
    kl_mean = sum(kls)/len(kls)
    kl_ctl.update(kl_mean.cpu().numpy(),ppo_config['num_rollouts'] * mpu.get_data_parallel_world_size())
    
    # 将 kl_mean 写入到log中
    writer = get_tensorboard_writer()
    score_mean = sum(reward_list)/len(reward_list)
    writer.add_scalar('ppo/kl_mean', kl_mean, args.ppo_learn_iteration)
    writer.add_scalar('ppo/kl_coef', kl_ctl.value, args.ppo_learn_iteration)
    writer.add_scalar('ppo/score_mean', score_mean, args.ppo_learn_iteration)
    # 将采样的文本写入到log 中
    record_generated_texts(prompts[:30],ret["generated_texts"][:30], reward_list[:30],args.ppo_learn_iteration)

    print('kl_mean',kl_mean,'kl_coef',kl_ctl.value)
    # print('ref_logprob_list',ref_logprob_list[:2])
    # print('old_rewards_list',old_rewards_list[:2])

    # print(sdfg)
    # print('non_score_rewards mean',sum(non_score_rewards_list)/len(non_score_rewards_list))

    input_tensors_list = [
      torch.cat((query_tensors,response_tensors),dim=-1) 
      for query_tensors, response_tensors in zip(query_tensors_list,response_tensors_list)
      ]
  
  # 将input_tensor_list同步到所有张量并行组的src rank
  if not input_tensors_list:
    input_tensors_list = [None]*len(prompts)
  
  input_tensors_list = broadcast_list_tensor_across_pp_src_group(
    input_tensors_list
  )

  # print_with_rank('input_tensors_list',input_tensors_list[0])
  _ppo_learn(
      model,
      ppo_config,
      forward_step_func,
      optimizer,
      lr_scheduler,
      micro_batch_size,
      timers, 
      input_tensors_list,
      old_logprob_list,old_values_list,
      old_rewards_list,
      query_tensors_list
    )

  # TODO 结尾设置kl系数
  return 


def generate_batched(
  prompts:list,
  unwrapped_model,
  max_new_tokens,
  tokenizer,
  eos_token_id,
  max_seq_len,
  batch_size=1,
  recompute=False,
  greedy=False,
  temperature=1,
  top_k=0,
  top_p=1,
  
  ):
  """
   以批次的方式进行采样
  Args:
      prompts (list): _description_
      unwrapped_model (_type_): _description_
      max_new_tokens (_type_): _description_
      eod (_type_): _description_
      batch_size (int, optional): _description_. Defaults to 1.
  """

  ret = {
    "query_tensors": [],
    "response_tensors": [],
    'values': [],
    'lm_logits': [],
    'generated_texts': []
  }
  prompt_num = len(prompts)
  if dist.get_rank() == 0:
    iter = tqdm(range(0,prompt_num,batch_size), total=prompt_num // batch_size)
  else:
    iter = range(0,prompt_num,batch_size)
  for batch_start in iter:
    prompts_batch = prompts[batch_start:(batch_start+batch_size)]
    # print_with_rank(f'batch start: {batch_start}',prompts_batch)
    r = generate_samples_ppo(
      unwrapped_model,
      prompts_batch,
      max_new_tokens,
      tokenizer,
      eos_token_id,
      max_seq_len,
      recompute=recompute,
      greedy=greedy,
      temperature=temperature,
      top_k=top_k,
      top_p=top_p,
    )
    torch.cuda.empty_cache()
    # 最后一个stage返回才不为None
    if r is None:
      ret = r 
    else:
      for k,v in ret.items():
        v.extend(r[k])
  # print_with_rank('finish generate')
  return ret

def compute_reward_with_kl_divergence(
  scores,
  ref_logprob_list,
  logprob_list,
  kl_ctl
  ):
  """

  Args:
      scores (_type_): _description_
      ref_logprob_list (_type_): _description_
      logprob_list (_type_): _description_
      response_tensor_len_list (_type_): _description_
  """
  rewards_list = []
  non_score_rewards_list = []
  kls = []
  for score,ref_logprob,logprob in zip(
    scores,ref_logprob_list,logprob_list):
    kls.append((logprob - ref_logprob).sum())
    # [response_size,]
    non_score_rewards = -kl_ctl.value * (logprob - ref_logprob
    )
    non_score_rewards_list.append(non_score_rewards)
    rewards = non_score_rewards.clone()
    # print('non_score_rewards',non_score_rewards,score)
    # 这里的需要特别注意，不能是rewards[-1] += score，因为rewards是2维的
    rewards[0][-1] += score  # 最后一个位置的reward加上句子的评分
    rewards_list.append(rewards)

  return rewards_list,non_score_rewards_list,kls


def _ppo_learn(
  model,
  ppo_config,
  forward_step_func,
  optimizer,
  lr_scheduler,
  micro_batch_size,
  timers,
  input_tensors_list,
  old_logprob_list,old_values_list,
  old_rewards_list,
  query_tensors_list,

  ):
  """根据经验进行一次ppo算法的学习

  Args:
      model (_type_): _description_
      input_tensors_list (_type_): _description_
      old_logprob_list (_type_): _description_
      old_values_list (_type_): _description_
      old_rewards_list (_type_): _description_
  """
  
  for model_module in model:
    model_module.train()
  args = get_args()
  update_num_microbatches(args.consumed_train_samples)
    
  # 不同类型的rank对应不同功能的数据集
  if is_tensor_model_parallel_src_rank():
    if mpu.is_pipeline_last_stage():
      rollout_dataset = RolloutDataset(
            input_tensors_list,
            old_logprob_list,
            old_values_list,
            old_rewards_list,
            query_tensors_list,
            input_tokens_only=False
          )
    else:
      rollout_dataset = RolloutDataset(
          input_tensors_list,
          old_logprob_list,
          old_values_list,
          old_rewards_list,
          query_tensors_list,
          input_tokens_only=True
        )
  else:
    rollout_dataset = None

  micro_batches = get_num_microbatches() 
  # micro_batch_size = argsmicro_batch_size
  # 先将 batch_num 传递到当前张量并行组
  if mpu.is_pipeline_last_stage():
    if is_last_stage_and_scr_rank():
      assert len(rollout_dataset) % (micro_batches*micro_batch_size) == 0, f'{len(rollout_dataset),{micro_batches},{micro_batch_size}}'
      batch_num = [math.ceil(len(rollout_dataset)/(micro_batches*micro_batch_size))]
      dist.broadcast_object_list(
        batch_num,
        src=mpu.get_tensor_model_parallel_src_rank(),
        group=mpu.get_tensor_model_parallel_group()
        )
    else:
      batch_num = [None]
      dist.broadcast_object_list(
        batch_num,
        src=mpu.get_tensor_model_parallel_src_rank(),
        group=mpu.get_tensor_model_parallel_group()
        )
    # 将batch num传递到其他stage中
    dist.broadcast_object_list(
        batch_num,
        src=mpu.get_pipeline_model_parallel_last_rank(),
        group=mpu.get_pipeline_model_parallel_group()
        )
  else:
    batch_num = [None]
    dist.broadcast_object_list(
        batch_num,
        src=mpu.get_pipeline_model_parallel_last_rank(),
        group=mpu.get_pipeline_model_parallel_group()
        )
  batch_num = batch_num[0]
  # print_with_rank('batch_numxxxx',batch_num)

  dist.barrier()

  all_stats = []
  # inner_step = 0
  for epoch_id in range(ppo_config['ppo_epochs']):
      if rollout_dataset is not None:
        # batch_num = math.ceil(len(rollout_dataset)/(micro_batches*micro_batch_size))
        # 这里shuffle等于false保证first stage 与last stage的数据是一致的
        # TODO 支持shuffle
        dataloader = iter(DataLoader(
          rollout_dataset,micro_batch_size,shuffle=False,num_workers=0,
          collate_fn=lambda x:x
          ))
      else:
        dataloader = None
      # 进行一次参数更新
      for _ in range(batch_num):

        # Set grad to zero.
        losses_reduced, skipped_iter, grad_norm, num_zeros_in_grad = ppo_train_step(
          model,forward_step_func,optimizer,dataloader,lr_scheduler,timers
          )
        args.ppo_iteration += 1
        if mpu.is_pipeline_last_stage():
          all_stats.append(losses_reduced[0])
        
        # print(sfgsfg)
        # if mpu.is_pipeline_last_stage():
        #   print('-*'*50)
        #   print('inner step',args.ppo_iteration)
        #   print('losses/policy_loss',losses_reduced[-1]['losses/policy_loss'])
        #   print('losses/value_loss',losses_reduced[-1]['losses/value_loss'])
        #   print('losses/total_loss',losses_reduced[-1]['losses/total_loss'])
        #   unwrapped_model = unwrap_model(model[0],
        #                             (torchDDP, LocalDDP, Float16Module))
        #   print('v_head weight',unwrapped_model.v_head[-1].weight.data[0,:5],unwrapped_model.v_head[-1].weight.grad.abs().mean())
        #   print('word embedding',unwrapped_model.word_embeddings.weight[0,:5],unwrapped_model.word_embeddings.weight.grad.abs().mean())
        #   print('policy/approx_kl',losses_reduced[-1]['policy/approx_kl'])
        #   all_stats.append(losses_reduced[0])
        #   print('-*'*50)
        #  = torch.LongTensor([[ 5828, 14574, 16623, 53332, 15227,    15,  3402, 19994,  9003, 28728, 52459,   373]])
        # with torch.no_grad():
        #   unwrapped_model(input_ids)

        # if args.ppo_iteration == 2:
    
        #   print(xdfb)
  
  # 汇总统计结果
  
  if mpu.is_pipeline_last_stage() \
    and mpu.get_data_parallel_rank() ==0 \
    and mpu.get_tensor_model_parallel_rank()==0: 
    # 仅仅对于一个rank才进行统计保存
    print('-*'*50)
    print('learn step',args.ppo_learn_iteration, 'inner step',args.ppo_iteration)
    print('losses/policy_loss',losses_reduced[-1]['losses/policy_loss'])
    print('losses/value_loss',losses_reduced[-1]['losses/value_loss'])
    print('losses/total_loss',losses_reduced[-1]['losses/total_loss'])
    unwrapped_model = unwrap_model(model[0],
                              (torchDDP, LocalDDP, Float16Module))
    print('v_head weight',unwrapped_model.v_head.layer.weight.data[0,:5],unwrapped_model.v_head.layer.weight.grad.abs().mean())
    print('word embedding',unwrapped_model.word_embeddings.weight[0,:5],unwrapped_model.word_embeddings.weight.grad.abs().mean())
    print('policy/approx_kl',losses_reduced[-1]['policy/approx_kl'])
    # all_stats.append(losses_reduced[0])
    print('-*'*50)
    # log to tensorboard
    record_step_stats(all_stats,args.ppo_learn_iteration)#,extra_kwargs={'kl_mean_ref':kl_mean_ref,'kl_coef':kl_ctl.value})
  
  args.ppo_learn_iteration += 1

  # if mpu.is_pipeline_last_stage():
  #   # print('stats_print',all_stats[0])
  #   print('losses/policy_loss',all_stats[-1]['losses/policy_loss'])
  #   print('losses/value_loss',all_stats[-1]['losses/value_loss'])
  #   print('policy/approx_kl',all_stats[-1]['policy/approx_kl'])


def ppo_train_step(model,forward_step_func,optimizer,dataloader,lr_scheduler,timers):

  # import signal 
  # signal.signal(signal.SIGALRM, sigalrm_handler)
  # signal.alarm(5)  # 5s 之后关闭程序

  # 打印模型参数
  # for n,p in unwrap_model(model[0],
  #                                   (torchDDP, LocalDDP, Float16Module)).named_parameters():
  #   print(n,p.shape,p.grad.sum() if p.grad is not None else None,p.requires_grad)
  
  for model_module in model:
    model_module.train()
  args = get_args()
  # 确定forward与backward函数
  if mpu.get_pipeline_model_parallel_world_size() > 1:
    if args.virtual_pipeline_model_parallel_size is not None:
      forward_backward_func = forward_backward_pipelining_with_interleaving
      assert get_num_microbatches() % args.pipeline_model_parallel_size == 0, \
          'number of microbatches is not divisible by pipeline-parallel ' \
          'size when using interleaved schedule'
    else:
      forward_backward_func = forward_backward_pipelining_without_interleaving
  else:
      forward_backward_func = forward_backward_no_pipelining
  # Set grad to zero.
  if not args.deepspeed:
    if args.DDP_impl == 'local' and args.use_contiguous_buffers_in_ddp:
      for partition in model:
        partition.zero_grad_buffer()
    else:
      optimizer.zero_grad()

  # optimizer.print_diff_btw_model_and_main()
  
  losses_reduced = forward_backward_func(
    forward_step_func,
    dataloader,
    model,
    optimizer,
    timers,
    forward_only=False
    )

  # print('-*'*100)
  # optimizer.print_diff_btw_model_and_main()
  
    # All-reduce if needed.
  if not args.deepspeed and args.DDP_impl == 'local':
    timers('backward-params-all-reduce').start()
    for model_module in model:
      model_module.allreduce_gradients()
    timers('backward-params-all-reduce').stop()

  # All-reduce word_embeddings' grad across first and last stages to ensure
  # that word_embeddings parameters stay in sync.
  # This should only run for models that support pipelined model parallelism
  # (BERT and GPT-2).
  timers('backward-embedding-all-reduce').start()
  if not args.deepspeed:
    if (mpu.is_pipeline_first_stage(ignore_virtual=True) or
        mpu.is_pipeline_last_stage(ignore_virtual=True)) and \
            mpu.get_pipeline_model_parallel_world_size() > 1:
      if mpu.is_pipeline_first_stage(ignore_virtual=True):
        unwrapped_model = model[0]
      elif mpu.is_pipeline_last_stage(ignore_virtual=True):
        unwrapped_model = model[-1]
      unwrapped_model = unwrap_model(unwrapped_model,
                                    (torchDDP, LocalDDP, Float16Module))

      if unwrapped_model.share_word_embeddings:
        word_embeddings_weight = unwrapped_model.word_embeddings_weight()
        if args.DDP_impl == 'local':
          grad = word_embeddings_weight.main_grad
        else:
          grad = word_embeddings_weight.grad
        torch.distributed.all_reduce(grad, group=mpu.get_embedding_group())
  timers('backward-embedding-all-reduce').stop()

  unwrapped_model = unwrap_model(model[0],
                                    (torchDDP, LocalDDP, Float16Module))
  # 打印模型参数
  # if mpu.is_pipeline_last_stage():
  # for n,p in unwrapped_model.named_parameters():
  #   if p.grad.abs().mean() > 0.0001:
  #     print('大梯度',n,p.shape,p.data.abs().mean(),p.grad.abs().mean() if p.grad is not None else None,p.requires_grad,p.is_leaf)
  
  # print(sfb)
  # Update parameters.
  timers('optimizer').start()
  if args.deepspeed:
    increment = get_num_microbatches() * \
                args.micro_batch_size * \
                args.data_parallel_size
    model[0].step(lr_kwargs={'increment': increment})
    update_successful = model[0].was_step_applied()
  else:
    # update_successful = True
    # skipped_iter = 0
    # grad_norm = None
    # num_zeros_in_grad = None
    # pass

    # def get_param_abs_mean():
    #   ret = []
    #   for n,p, in unwrapped_model.named_parameters():
    #     ret.append((n,p.data.clone(),p.grad.data.clone()))
    #   return ret 


    # before_update_param_abs_mean = get_param_abs_mean()
    update_successful, grad_norm, num_zeros_in_grad = optimizer.step()
    # after_update_param_abs_mean = get_param_abs_mean() 
    # diff = [(n,(m1-m2).abs().max(),g1.abs().max(),g2.max().mean())for (n,m1,g1),(n,m2,g2) in zip(before_update_param_abs_mean,after_update_param_abs_mean)]
    # for n,d ,g1,g2 in diff:
    #   print(f'update diff, {n},{d},before grad: {g1}, after grad: {g2}')
    
    # print('grad_norm',grad_norm)
    # print('num_zeros_in_grad',num_zeros_in_grad)
    # print(fg)
  timers('optimizer').stop()
  # Update learning rate.
  if args.deepspeed:
    skipped_iter = 0
    grad_norm = None
    num_zeros_in_grad = None
  else:
    if update_successful:
      increment = get_num_microbatches() * \
                  args.micro_batch_size * \
                  args.data_parallel_size
      if lr_scheduler is not None:
        lr_scheduler.step(increment=increment)
      skipped_iter = 0
    else:
      print_rank_0(f'skip update 。。。')
      skipped_iter = 1

    if mpu.is_pipeline_last_stage(ignore_virtual=True):
      # Average loss across microbatches.
      # loss_reduced = {}
      # for key in losses_reduced[0]:
      #   losses_reduced_for_key = [x[key] for x in losses_reduced]
      #   loss_reduced[key] = sum(losses_reduced_for_key) / len(
      #       losses_reduced_for_key)
      return losses_reduced, skipped_iter, grad_norm, num_zeros_in_grad
  return {}, skipped_iter, grad_norm, num_zeros_in_grad
  






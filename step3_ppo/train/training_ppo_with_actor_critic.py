""" 
    PPO trainer for training ActorCritic model
"""
import math
import sys
import time
import json
import numpy as np

_TRAIN_START_TIME = time.time()

import torch
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP
from torch.utils.data import DataLoader

from megatron import get_args
from megatron import get_timers
from megatron import get_tensorboard_writer
from megatron import get_num_microbatches
from megatron import update_num_microbatches
from megatron import mpu
from megatron import get_tokenizer
from megatron import print_rank_0, print_with_rank
from megatron.checkpointing import save_checkpoint
from megatron.model.module import Float16Module
from megatron.initialize import initialize_megatron
from megatron.model.distributed import DistributedDataParallel as LocalDDP
from megatron.utils import unwrap_model, found_kill_switch
from megatron.schedules import forward_backward_no_pipelining
from megatron.schedules import forward_backward_pipelining_without_interleaving
from megatron.schedules import forward_backward_pipelining_with_interleaving
from megatron.training import setup_model_and_optimizer, print_datetime, save_checkpoint_and_time
import torch.distributed as dist
from megatron.data.data_samplers import MegatronPretrainingRandomSampler
from step3_ppo.train.ppo_dataset import RolloutDataset
import os

WANDB_PADDING = -1


def is_last_stage_and_scr_rank():
  """ Check if it is in the last_pipeline_stage and the first rank of the tensor parallel group
  """
  return  mpu.is_pipeline_last_stage() and \
      dist.get_rank()==mpu.get_tensor_model_parallel_src_rank()


def is_first_stage_and_scr_rank():
  """ Check if it is in first pipeline stage and the first rank of the tensor parallel group
  """
  return mpu.is_pipeline_first_stage() and \
      dist.get_rank()==mpu.get_tensor_model_parallel_src_rank()


def is_tensor_model_parallel_src_rank():
  return dist.get_rank() == mpu.get_tensor_model_parallel_src_rank()


def broadcast_prompts_across_pp_tp_group(prompts: list):
  """
    Broadcast the object list to all pp stages and tp stages
     1. broadcast them from last stage to all other stages
     2. broadcast them from first tensor rank to all other tensor ranks
  """
  assert isinstance(prompts, list) and isinstance(prompts[0], str), prompts
  dist.broadcast_object_list(prompts,
                             src=mpu.get_pipeline_model_parallel_last_rank(),
                             group=mpu.get_pipeline_model_parallel_group())

  dist.broadcast_object_list(prompts,
                             src=mpu.get_tensor_model_parallel_src_rank(),
                             group=mpu.get_tensor_model_parallel_group())

  return prompts


def broadcast_list_tensor_across_pp_src_group(input_tensors_list: list):

  pmp_last_rank = mpu.get_pipeline_model_parallel_last_rank()
  if is_last_stage_and_scr_rank():
    assert dist.get_rank() == pmp_last_rank
    dist.broadcast_object_list(
        [input_tensors for input_tensors in input_tensors_list],
        src=pmp_last_rank,
        group=mpu.get_pipeline_model_parallel_group())
  elif mpu.get_tensor_model_parallel_src_rank() == dist.get_rank():

    dist.broadcast_object_list(input_tensors_list,
                               src=pmp_last_rank,
                               group=mpu.get_pipeline_model_parallel_group())
    input_tensors_list = [
        input_tensors for input_tensors in input_tensors_list
    ]
  return input_tensors_list


def collect_logs(logs, key, value):
  logs.append({'name': key, 'value': value})


def write_tensorboard_logs_tp(logs,
                              ppo_learn_iteration,
                              tensor_parallel_model_size=1):
  """ Write tensorboard for TP > 1
  """
  if tensor_parallel_model_size == 1: return

  if mpu.is_pipeline_last_stage():
    print_with_rank("> tensorboard writing ... ")
    if mpu.is_last_stage_and_scr_rank():
      log_num = [len(logs)]
      dist.broadcast_object_list(log_num,
                                 src=mpu.get_tensor_model_parallel_src_rank(),
                                 group=mpu.get_tensor_model_parallel_group())
    else:
      log_num = [None]
      dist.broadcast_object_list(log_num,
                                 src=mpu.get_tensor_model_parallel_src_rank(),
                                 group=mpu.get_tensor_model_parallel_group())
    # broadcast tensorboard log data
    if mpu.is_last_stage_and_scr_rank():
      dist.broadcast_object_list(logs,
                                 src=mpu.get_tensor_model_parallel_src_rank(),
                                 group=mpu.get_tensor_model_parallel_group())
    else:
      logs = [None] * log_num[0]
      dist.broadcast_object_list(logs,
                                 src=mpu.get_tensor_model_parallel_src_rank(),
                                 group=mpu.get_tensor_model_parallel_group())
      # write tensorboard log
      record_tensorboard_logs(logs, ppo_learn_iteration)


def record_tensorboard_logs(logs, iteration):
  writer = get_tensorboard_writer()
  if not writer: pass
  else:
    for log in logs:
      log_name = log['name']
      if 'distribution' in log_name:
        writer.add_histogram(log_name, np.array(log['value']), iteration)
      elif 'generate_sample' in log_name:
        prompts = log['value']['prompts']
        generated_texts = log['value']['generated_texts']
        rewards = log['value']['rewards']
        tag_prefix = log['value']['tag_prefix']
        record_generated_texts(prompts, generated_texts, rewards, iteration,
                               tag_prefix)
      else:
        writer.add_scalar(log_name, log['value'], iteration)


def record_generated_texts(prompts,
                           generated_texts,
                           rewards,
                           iteration,
                           tag_prefix='ppo/text_generate_sample_step_'):
  writer = get_tensorboard_writer()
  markdown_table = """
  | prompt | response | score |
  | :----: | :----: | :----: |"""
  for prompt, generated_text, reward in zip(prompts, generated_texts, rewards):
    prompt = prompt.replace("\n", "<br />")
    generated_text = generated_text.replace("\n", "<br />").replace('|', '\|')
    row = '\n' + f'| {prompt} | {generated_text} | {round(reward,3)}|'
    markdown_table += row
  writer.add_text(tag_prefix + str(iteration), markdown_table, iteration)


def record_train_args(args):
  writer = get_tensorboard_writer()
  if writer is None:
    return
  markdown_table = """
  | key | value |
  | :----: | :----: |"""
  for arg in vars(args):
    key = arg
    value = getattr(args, arg)
    row = '\n' + f'| {key} | {value} |'
    markdown_table += row
  writer.add_text('train/args', markdown_table, 0)


def record_step_stats(all_stats: list, iteration, extra_kwargs=None):

  def average(scalars: list):
    return sum(scalars) / len(scalars)

  stats = {}
  for k in all_stats[0].keys():
    stats[k] = average([l[k] for l in all_stats])

  # log to tensorboard
  writer = get_tensorboard_writer()
  if writer is not None:
    for k, v in stats.items():
      writer.add_scalar(k, v, iteration)

    if extra_kwargs is not None:
      assert isinstance(extra_kwargs, dict)
      for k, v in extra_kwargs.items():
        writer.add_scalar(k, v, iteration)
  return stats


def ppo_train(prompts_dataset_provider,
              actor_critic_model_provider,
              forward_step_func,
              reward_fn,
              ref_model_fn,
              kl_ctl_provider,
              extra_args_provider=None,
              param_groups_provider=None,
              learning_rate_scheduler_provider=None,
              ptx_dataset_provider=None,
              ptx_forward_step_func=None,
              args_defaults={}):
  """
      Train program for ppo algorithm.
      This function will run the followings in the order provided:
          1) initialize Megatron.
          2) setup model, optimizer and lr schedule using the model_provider.
          3) call train_val_test_data_provider to get train/val/test datasets.
          4) train the model.
      prompts_iter_provider: a function that provide train/val/test prompts
      model_provider: a function that provide actor_critic_model,
      reward_fn: a method to calculate reward of the sampling.
      ref_model_fn: a ref model to compute logprobs of the sampling.
      extra_args_provider: a function that takes a parser and adds arguments
              to it. It is used for programs to add their own arguments.
      args_defaults: a dictionary from argument-name to argument-value. It
              to set already parse arguments.
  """

  # megatron initialization
  initialize_megatron(extra_args_provider=extra_args_provider,
                      args_defaults=args_defaults)

  args = get_args()
  record_train_args(args)

  if args.ptx_dataset_path and os.path.exists(args.ptx_dataset_path):
    args.use_ptx = True
  else:
    args.use_ptx = False

  global _TRAIN_START_TIME
  start_time_tensor = torch.cuda.FloatTensor([_TRAIN_START_TIME])
  torch.distributed.all_reduce(start_time_tensor,
                               op=torch.distributed.ReduceOp.MIN)
  _TRAIN_START_TIME = start_time_tensor.item()
  print_rank_0('time to initialize megatron (seconds): {:.3f}'.format(
      time.time() - _TRAIN_START_TIME))

  # steup model，optimizer, lr_scheduler
  model, optimizer, lr_scheduler = setup_model_and_optimizer(
      actor_critic_model_provider,
      param_groups_provider=param_groups_provider,
      learning_rate_scheduler_provider=learning_rate_scheduler_provider)

  print_datetime('after megatron is initialized')
  timers = get_timers()

  iteration = 0
  unwrapped_model = unwrap_model(model[0], (torchDDP, LocalDDP, Float16Module))
  ppo_config = unwrapped_model.ppo_config
  num_rollouts = ppo_config['num_rollouts']

  # ppo args
  args.temperature = ppo_config['gen_kwargs']['temperature']
  args.top_k = ppo_config['gen_kwargs']['top_k']
  args.top_p = ppo_config['gen_kwargs']['top_p']
  args.greedy = False
  args.recompute = False
  args.ppo_iteration = 0
  args.ppo_learn_iteration = 0
  args.ppo_config = ppo_config
  print_rank_0(json.dumps(ppo_config, ensure_ascii=False, indent=2))

  # ppo train dataloader
  total_prompt_sample_num = args.train_iters * num_rollouts
  prompt_dataset, eval_prompts = prompts_dataset_provider()
  print_rank_0('prompt train epoch: ',
               total_prompt_sample_num / len(prompt_dataset))
  prompt_sampler = MegatronPretrainingRandomSampler(
      total_samples=total_prompt_sample_num,
      consumed_samples=0,
      micro_batch_size=num_rollouts,
      data_parallel_rank=mpu.get_data_parallel_rank(),
      data_parallel_size=mpu.get_data_parallel_world_size())
  prompt_dataloader = torch.utils.data.DataLoader(
      prompt_dataset,
      batch_sampler=prompt_sampler,
      shuffle=False,
      num_workers=args.num_workers,
      generator=torch.Generator().manual_seed(args.seed),
      collate_fn=lambda x: x)

  # ptx dataloader
  if args.use_ptx:
    ptx_dataset = ptx_dataset_provider()
    ptx_sample_num = args.train_iters * num_rollouts * args.ppo_config[
        'ppo_epochs']
    print_rank_0('ptx train epoch: ', ptx_sample_num / len(ptx_dataset))
    ptx_batch_sampler = MegatronPretrainingRandomSampler(
        total_samples=ptx_sample_num,
        consumed_samples=0,
        micro_batch_size=args.micro_batch_size,
        data_parallel_rank=mpu.get_data_parallel_rank(),
        data_parallel_size=mpu.get_data_parallel_world_size())
    ptx_dataloader = iter(
        torch.utils.data.DataLoader(ptx_dataset,
                                    batch_sampler=ptx_batch_sampler,
                                    shuffle=False,
                                    num_workers=args.num_workers,
                                    generator=torch.Generator().manual_seed(
                                        args.seed),
                                    collate_fn=lambda x: x))
  else:
    ptx_dataloader = None

  if ptx_dataloader is not None: print_rank_0('using ptx loss')

  # kl controller
  kl_ctl = kl_ctl_provider(ppo_config)

  tokenizer = get_tokenizer()

  # ppp training
  while iteration < args.train_iters:
    # 特殊情况下关闭训练
    if found_kill_switch():
      save_checkpoint_and_time(iteration, model, optimizer, lr_scheduler)
      print_datetime(
          f"Detected kill switch at {args.kill_switch_path}. Exiting")
      sys.exit()

    for prompts in prompt_dataloader:

      # training is done
      if iteration >= args.train_iters:
        print_rank_0('train finish')
        save_checkpoint(iteration, model, optimizer, lr_scheduler)
        return

      # give up the last rollout if the number of prompts is not num_rollouts
      # if len(prompts) != num_rollouts: break
      # add the frequency of prompts
      prompts = prompts * args.ppo_sampling_times

      # one rollout of ppo training
      ppo_learn(prompts,
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
                kl_ctl,
                ptx_dataloader=ptx_dataloader,
                ptx_forward_step_func=ptx_forward_step_func,
                eval_prompts=eval_prompts)

      # save checkpoint every save_inerval iteration
      if iteration != 0 and iteration % args.save_interval == 0:
        print_rank_0(f'Saving the checkpoint: {iteration}')
        save_checkpoint(iteration, model, optimizer, lr_scheduler)

      iteration += 1


def ppo_eval(unwrapped_model, prompts, reward_fn):
  """ Evaluate the actor model
    1. generate text responses for eval prompts
    2. generate the corresponding reward scores
  """
  if prompts is None:
    return None, None
  eval_data = unwrapped_model.eval_prompts(prompts)
  generated_texts = eval_data['generated_texts']
  if generated_texts:
    texts = [
        prompt + generated_text
        for prompt, generated_text in zip(prompts, generated_texts)
    ]
    rewards_list = reward_fn(texts)
  else:
    rewards_list = []
  return generated_texts, rewards_list


def ppo_learn(prompts,
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
              generate_kwargs: dict,
              ppo_config,
              kl_ctl,
              ptx_dataloader=None,
              ptx_forward_step_func=None,
              eval_prompts=None):
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

  input_tensors_list = []
  query_tensors_list = []
  response_tensors_list = []
  old_logprob_list = []
  old_values_list = []
  old_rewards_list = []

  # synchronize the prompts
  prompts = broadcast_prompts_across_pp_tp_group(prompts)
  arigin_checkpoint_activations = args.checkpoint_activations
  unwrapped_model.actor.args.checkpoint_activations = False
  unwrapped_model.critic.args.checkpoint_activations = False

  # iteract with the environment
  # 1). sampling (actor, critic) 2). ref_logprobs (reference model)
  # 3). rewards  (reward model)
  experience = unwrapped_model.make_experience(prompts, reward_fn,
                                               ref_model_fn, kl_ctl)

  # evaluation
  if eval_prompts is not None and  \
    args.ppo_learn_iteration % args.eval_interval_iteration == 0:
    print_rank_0('eval......')
    eval_generated_texts, eval_rewards = ppo_eval(unwrapped_model,
                                                  eval_prompts, reward_fn)

  unwrapped_model.actor.args.checkpoint_activations = arigin_checkpoint_activations
  unwrapped_model.critic.args.checkpoint_activations = arigin_checkpoint_activations

  query_tensors_list = experience['query_tensors_list']
  response_tensors_list = experience['response_tensors_list']
  old_logprob_list = experience['old_logprob_list']
  old_values_list = experience['old_values_list']
  old_rewards_list = experience['old_rewards_list']
  timing = experience['timing']
  kl_mean = experience['kl_mean']
  kl_token_mean = experience['kl_token_mean']
  reward_list = experience['reward_list']
  generated_texts = experience['generated_texts']

  # tensorboard logging
  logs = []
  if mpu.is_last_stage_and_scr_rank():
    # update kl
    kl_ctl.update(kl_mean.cpu().numpy())
    writer = get_tensorboard_writer()
    score_mean = sum(reward_list) / len(reward_list)
    score_var = np.array(reward_list).var().tolist()

    if writer:
      writer.add_scalar('ppo/kl_mean', kl_mean, args.ppo_learn_iteration)
      writer.add_scalar('ppo/kl_token_mean', kl_token_mean,
                        args.ppo_learn_iteration)
      writer.add_scalar('ppo/kl_coef', kl_ctl.value, args.ppo_learn_iteration)
      writer.add_scalar('reward/score_var', score_var,
                        args.ppo_learn_iteration)

      # write the histogram of training rewards
      writer.add_histogram('reward/train_distrubution', np.array(reward_list),
                           args.ppo_learn_iteration)
      writer.add_scalar('ppo/train_score_mean', score_mean,
                        args.ppo_learn_iteration)

      # write the time
      for k, v in timing.items():
        writer.add_scalar(k, v, args.ppo_learn_iteration)

      # write the generated texts for training samples
      record_generated_texts(prompts[:30],
                             generated_texts[:30],
                             reward_list[:30],
                             args.ppo_learn_iteration,
                             tag_prefix='ppo_train/text_generate_sample_step_')

      if args.ppo_learn_iteration % args.eval_interval_iteration == 0 and \
        eval_prompts is not None:
        # write the generated texts for eval samples
        record_generated_texts(
            eval_prompts,
            eval_generated_texts,
            eval_rewards,
            args.ppo_learn_iteration,
            tag_prefix='ppo_eval/text_generate_sample_step_')
        # write the histogram of eval rewards
        writer.add_histogram('reward/eval_distribution',
                             np.array(eval_rewards), args.ppo_learn_iteration)
        writer.add_scalar('ppo/eval_score_mean',
                          np.array(eval_rewards).mean(),
                          args.ppo_learn_iteration)

    # collect tensorboard logs
    collect_logs(logs, 'ppo/kl_mean', kl_mean)
    collect_logs(logs, 'ppo/kl_token_mean', kl_token_mean)
    collect_logs(logs, 'ppo/train_score_mean', score_mean)
    collect_logs(logs, 'reward/score_var', score_var)
    collect_logs(logs, 'reward/train_distribution', reward_list)
    collect_logs(
        logs, 'ppo_train/text_generate_sample', {
            'prompts': prompts[:30],
            'generated_texts': generated_texts[:30],
            'rewards': reward_list[:30],
            'tag_prefix': 'ppo_train/text_generate_sample_step_'
        })
    collect_logs(
        logs, 'ppo_eval/text_generate_sample', {
            'prompts': eval_prompts,
            'generated_texts': eval_generated_texts,
            'rewards': eval_rewards,
            'tag_prefix': 'ppo_eval/text_generate_sample_step_'
        })
    collect_logs(logs, 'ppo/eval_score_mean', np.array(eval_rewards).mean())
    collect_logs(logs, 'reward/eval_distribution', eval_rewards)
    print_with_rank('kl_mean', kl_mean, 'kl_token_mean', kl_token_mean,
                    'kl_coef', kl_ctl.value, 'score_mean', score_mean,
                    'score_var', score_var)

    input_tensors_list = [
        torch.cat((query_tensors, response_tensors),
                  dim=-1) for query_tensors, response_tensors in zip(
                      query_tensors_list, response_tensors_list)
    ]

  # broadcast the input_tensors to all tensor ranks
  if not input_tensors_list: input_tensors_list = [None] * len(prompts)
  input_tensors_list = broadcast_list_tensor_across_pp_src_group(
      input_tensors_list)

  ppo_learn_start_time = time.time()
  # one step of PPO training
  _ppo_learn(model,
             ppo_config,
             forward_step_func,
             optimizer,
             lr_scheduler,
             micro_batch_size,
             timers,
             input_tensors_list,
             old_logprob_list,
             old_values_list,
             old_rewards_list,
             query_tensors_list,
             ptx_dataloader=ptx_dataloader,
             ptx_forward_step_func=ptx_forward_step_func)

  if lr_scheduler is not None:
    lr_scheduler.step()

  if mpu.is_last_stage_and_scr_rank():
    if writer:
      writer.add_scalar('timing/ppo/ppo_learning_time',
                        time.time() - ppo_learn_start_time,
                        args.ppo_learn_iteration)
      if lr_scheduler is not None:
        for lr_index, lr in enumerate(lr_scheduler.get_lr()):
          writer.add_scalar(f'learning_rate/ppo_lr_{lr_index}', lr,
                            args.ppo_learn_iteration)
    collect_logs(logs, 'timing/ppo/ppo_learning_time',
                 time.time() - ppo_learn_start_time)

  # write the collected logs to the tensorboard
  write_tensorboard_logs_tp(logs, args.ppo_learn_iteration - 1,
                            args.tensor_model_parallel_size)

  return


def _ppo_learn(model,
               ppo_config,
               forward_step_func,
               optimizer,
               lr_scheduler,
               micro_batch_size,
               timers,
               input_tensors_list,
               old_logprob_list,
               old_values_list,
               old_rewards_list,
               query_tensors_list,
               ptx_dataloader=None,
               ptx_forward_step_func=None):
  """ PPO training for one rollout

  Args:
      model (_type_): _description_
      input_tensors_list (_type_): _description_
      old_logprob_list (_type_): _description_
      old_values_list (_type_): _description_
      old_rewards_list (_type_): _description_
  """
  args = get_args()

  for model_module in model:
    model_module.train()
  update_num_microbatches(args.consumed_train_samples)

  # 不同类型的rank对应不同功能的数据集
  if is_tensor_model_parallel_src_rank():
    if mpu.is_pipeline_last_stage():
      # last stage dataset
      rollout_dataset = RolloutDataset(input_tensors_list,
                                       old_logprob_list,
                                       old_values_list,
                                       old_rewards_list,
                                       query_tensors_list,
                                       input_tokens_only=False)
    else:
      # other stage dataset
      rollout_dataset = RolloutDataset(input_tensors_list,
                                       old_logprob_list,
                                       old_values_list,
                                       old_rewards_list,
                                       query_tensors_list,
                                       input_tokens_only=True)
  else:
    rollout_dataset = None

  micro_batches = get_num_microbatches()

  # 先将 batch_num 传递到当前张量并行组
  if mpu.is_pipeline_last_stage():

    # last pipeline stage: first tensor rank -> other tensor ranks
    if is_last_stage_and_scr_rank():
      # first tensor rank
      assert len(rollout_dataset) % (micro_batches*micro_batch_size) == 0, \
            f'{len(rollout_dataset),{micro_batches},{micro_batch_size}}'
      batch_num = [
          math.ceil(len(rollout_dataset) / (micro_batches * micro_batch_size))
      ]
      dist.broadcast_object_list(batch_num,
                                 src=mpu.get_tensor_model_parallel_src_rank(),
                                 group=mpu.get_tensor_model_parallel_group())
    else:
      # other tensor ranks
      batch_num = [None]
      dist.broadcast_object_list(batch_num,
                                 src=mpu.get_tensor_model_parallel_src_rank(),
                                 group=mpu.get_tensor_model_parallel_group())

    # last pipeline stage,first tensor rank -> other pipeline stages, first tensor ranks
    dist.broadcast_object_list(batch_num,
                               src=mpu.get_pipeline_model_parallel_last_rank(),
                               group=mpu.get_pipeline_model_parallel_group())

  else:
    # other pipeline stages
    batch_num = [None]
    dist.broadcast_object_list(batch_num,
                               src=mpu.get_pipeline_model_parallel_last_rank(),
                               group=mpu.get_pipeline_model_parallel_group())

  batch_num = batch_num[0]

  dist.barrier()

  all_stats = []
  start_learn_time = time.time()

  # inner PPO epochs after each rollout
  for epoch_id in range(ppo_config['ppo_epochs']):

    torch.cuda.empty_cache()  # cleanup cuda cache

    if rollout_dataset is not None:
      # create a rollout dataset
      print_rank_0(f"> Number of sampling in the current rollout: {len(rollout_dataset)}")
      # TODO 支持shuffle
      dataloader = iter(
          DataLoader(rollout_dataset,
                     micro_batch_size,
                     shuffle=False,
                     num_workers=0,
                     collate_fn=lambda x: x))
    else:
      dataloader = None

    # update weights of actor and critic
    for _ in range(batch_num):

      torch.cuda.empty_cache()
      # one step of PPO training
      losses_reduced, skipped_iter, grad_norm, num_zeros_in_grad = ppo_train_step(
          model,
          forward_step_func,
          optimizer,
          dataloader,
          lr_scheduler,
          timers,
          ptx_dataloader=ptx_dataloader,
          ptx_forward_step_func=ptx_forward_step_func)
      args.ppo_iteration += 1
      if mpu.is_pipeline_last_stage():
        all_stats.append(losses_reduced)

  logs = []
  if mpu.is_pipeline_last_stage() \
    and mpu.get_data_parallel_rank() ==0 \
    and mpu.get_tensor_model_parallel_rank()==0:

    # ppo losses
    ppo_all_stats = sum([i["ppo_losses_reduced"] for i in all_stats], [])
    ppo_stats = record_step_stats(ppo_all_stats, args.ppo_learn_iteration)
    for k, v in ppo_stats.items():
      collect_logs(logs, k, v)  # collect tensorboard logs

    print('-*' * 50)
    print('learn step', args.ppo_learn_iteration, 'inner step',
          args.ppo_iteration)
    print('losses/policy_loss', ppo_stats['losses/policy_loss'])
    print('losses/value_loss', ppo_stats['losses/value_loss'])
    print('losses/total_loss', ppo_stats['losses/total_loss'])
    print('policy/approx_kl', ppo_stats['policy/approx_kl'])

    # ptx losses
    if "ptx_losses_reduced" in all_stats[0]:
      ptx_all_stats = sum([i["ptx_losses_reduced"] for i in all_stats], [])
      ptx_stats = record_step_stats(ptx_all_stats, args.ppo_learn_iteration)
      for k, v in ptx_stats.items():
        collect_logs(logs, k, v)
      print('losses/ptx_loss', ptx_stats['losses/ptx_loss'])

    print(f'learn time: {time.time()-start_learn_time}')
    print('-*' * 50)

  # write tensorboard logs
  write_tensorboard_logs_tp(logs, args.ppo_learn_iteration,
                            args.tensor_model_parallel_size)

  args.ppo_learn_iteration += 1

  # if mpu.is_pipeline_last_stage():
  #   # print('stats_print',all_stats[0])
  #   print('losses/policy_loss',all_stats[-1]['losses/policy_loss'])
  #   print('losses/value_loss',all_stats[-1]['losses/value_loss'])
  #   print('policy/approx_kl',all_stats[-1]['policy/approx_kl'])


def ppo_train_step(model,
                   forward_step_func,
                   optimizer,
                   dataloader,
                   lr_scheduler,
                   timers,
                   ptx_dataloader=None,
                   ptx_forward_step_func=None):

  for model_module in model:
    model_module.train()
  args = get_args()

  # forward and backward function
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

  # PPO training
  losses_reduced_dict = {}
  ppo_losses_reduced = forward_backward_func(forward_step_func,
                                             dataloader,
                                             model,
                                             optimizer,
                                             timers,
                                             forward_only=False)
  losses_reduced_dict['ppo_losses_reduced'] = ppo_losses_reduced

  # ptx train
  if ptx_dataloader is not None:
    ptx_losses_reduced = forward_backward_func(ptx_forward_step_func,
                                               ptx_dataloader,
                                               model,
                                               optimizer,
                                               timers,
                                               forward_only=False)
    losses_reduced_dict['ptx_losses_reduced'] = ptx_losses_reduced

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
    # first and last stage
    if (mpu.is_pipeline_first_stage(ignore_virtual=True) or
        mpu.is_pipeline_last_stage(ignore_virtual=True)) and \
            mpu.get_pipeline_model_parallel_world_size() > 1:
      if mpu.is_pipeline_first_stage(ignore_virtual=True):
        unwrapped_model = model[0]
      elif mpu.is_pipeline_last_stage(ignore_virtual=True):
        unwrapped_model = model[-1]

      unwrapped_model = unwrap_model(unwrapped_model,
                                     (torchDDP, LocalDDP, Float16Module))

      if unwrapped_model.actor.share_word_embeddings:
        word_embeddings_weight = unwrapped_model.actor.word_embeddings_weight()
        if args.DDP_impl == 'local':
          grad = word_embeddings_weight.main_grad
        else:
          grad = word_embeddings_weight.grad
        torch.distributed.all_reduce(grad, group=mpu.get_embedding_group())
  timers('backward-embedding-all-reduce').stop()

  unwrapped_model = unwrap_model(model[0], (torchDDP, LocalDDP, Float16Module))
  # Update parameters.
  timers('optimizer').start()
  if args.deepspeed:
    increment = get_num_microbatches() * \
                args.micro_batch_size * \
                args.data_parallel_size
    model[0].step(lr_kwargs={'increment': increment})
    update_successful = model[0].was_step_applied()
  else:
    update_successful, grad_norm, num_zeros_in_grad = optimizer.step()
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
      skipped_iter = 0
    else:
      print_rank_0(f'skip update 。。。')
      skipped_iter = 1

    if mpu.is_pipeline_last_stage(ignore_virtual=True):
      return losses_reduced_dict, skipped_iter, grad_norm, num_zeros_in_grad

  return {}, skipped_iter, grad_norm, num_zeros_in_grad

# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""ActorCritic Model based on GPT2 model"""

from typing import Any, Dict, List, MutableMapping, Tuple, Union
from functools import partial
import torch
import torch.nn.init as init
from torch import nn
from tqdm import tqdm
import random
import numpy as np
import json
import time


from megatron import get_args
from megatron import mpu,print_rank_0, print_with_rank
from megatron import get_tokenizer
import torch.distributed as dist
from megatron.enums import AttnMaskType
from .module import MegatronModule, fp32_to_float16
import math,copy
from torch.nn import functional as F

from .language_model import parallel_lm_logits
from .language_model import get_language_model
from .utils import init_method_normal
from .utils import scaled_init_method_normal

from deepspeed.pipe import PipelineModule, LayerSpec, TiedLayerSpec
from megatron.model.fused_layer_norm import MixedFusedLayerNorm as LayerNorm
from megatron.model.module import float16_to_fp32
from .language_model import EmbeddingPipe
from .transformer import ParallelTransformerLayerPipe
from .gpt_model import GPTModel,post_language_model_processing
from .gpt_reward_model import GPTModelCritic
from megatron.text_generation_ppo import generate_samples as generate_samples_ppo,generate_samples_with_pipeline
from megatron import get_tensorboard_writer

class ParallelLinear(nn.Module):
  """Linear layer parallelized over the longer dimension."""

  def __init__(
      self,
      in_size: int,
      out_size: int,
      init_method=init.xavier_normal_,
      bias=True,
      gather_output=True,
      input_is_parallel=False,
      args=None
      ):
    super().__init__()

    if in_size < out_size:
      self.layer = mpu.ColumnParallelLinear(in_size,
                                            out_size,
                                            bias=bias,
                                            gather_output=gather_output,
                                            init_method=init_method,
                                            skip_bias_add=False,
                                            args=args
                                            )
    else:
      self.layer = mpu.RowParallelLinear(
          in_size,
          out_size,
          input_is_parallel=input_is_parallel,
          init_method=init_method,
          skip_bias_add=False,
          bias=bias,
          args=args
      )

  def forward(self, x):
    output, bias = self.layer(x)
    if bias is not None:
      return output + bias
    return output

import numpy as np
class RunningMoments:
    def __init__(self,):
        """
        Calculates the running mean and standard deviation of a data stream. Modified version of
        https://github.com/DLR-RM/stable-baselines3/blob/a6f5049a99a4c21a6f0bcce458ca3306cef310e0/stable_baselines3/common/running_mean_std.py
        """
        self.mean = 0
        self.std = 1
        self.var = 1
        self.count = 1e-24

    def update(self, xs: float) -> Tuple[float, float]:
        """
        Updates running moments from batch's moments computed across ranks
        """
        xs_count = len(xs)
        # xs_var, xs_mean = torch.var_mean(xs, unbiased=False)
        xs_var, xs_mean = np.std(xs), np.mean(xs)
        # xs_mean, xs_var = xs_mean.float(), xs_var.float()

        delta = xs_mean - self.mean         # mean difference
        tot_count = self.count + xs_count   # total count

        new_sum = xs_var * xs_count
        # correct old_sum deviation accounting for the new mean
        old_sum = self.var * self.count + delta**2 * self.count * xs_count / tot_count
        tot_sum = old_sum + new_sum

        self.mean += delta * xs_count / tot_count
        self.var = tot_sum / tot_count
        self.std = np.sqrt(self.var * tot_count / (tot_count - 1))
        self.count = tot_count
        
        return self.mean, self.std
        # return xs_mean, (xs_var * xs_count / (xs_count - 1)).float().sqrt()
        # return xs_mean.item(), (xs_var * xs_count / (xs_count - 1)).float().sqrt().item()

class ActorCriticModel(torch.nn.Module):
  """
  ppo实现,区别actor, critic方法
  Args:
      torch (_type_): _description_
  """
  default_ppo_config = dict(
      num_rollouts=128,
      ppo_epochs=4,
      init_kl_coef=0.2,
      target=6,
      horizon=10000,
      gamma=1,
      lam=0.95,
      cliprange=0.2,
      cliprange_value=0.2,
      vf_coef=0.1,
      scale_reward="ignored",
      ref_mean=None,
      adap_kl_ctrl=True,
      ref_std=None,
      cliprange_reward=0.2,
      reward_normalize=False,
      gen_kwargs=dict(max_new_tokens=40,
                      top_k=0,
                      top_p=1,
                      do_sample=True,
                      temperature=1),
  )

  def __init__(
    self,
    actor_args,
    critic_args,
    args,
    num_tokentypes=0,
    parallel_output=True,
    pre_process=True,
    post_process=True,
    prefix_lm=False,
    ) -> None:
    """

    Args:
        actor_args: actor 模型需要的参数
        critic_args: critic模型需要的参数
        args (_type_): 通用的参数
    """
    super().__init__()
    self.pre_process = pre_process
    self.post_process = post_process
    self.actor_args = actor_args
    self.critic_args = critic_args
    self.args = args

    # read ppo config
    with open(args.ppo_config_file) as f:
      ppo_config = json.load(f)
      self.ppo_config = self.default_ppo_config.copy()
      self.ppo_config.update(**ppo_config)
    
    # tokenizer
    self.tokenizer = get_tokenizer()

    # actor model
    self.actor = GPTModel(
      num_tokentypes=num_tokentypes,
      parallel_output=parallel_output,
      pre_process=pre_process,
      post_process=post_process,
      prefix_lm=prefix_lm,
      args = actor_args
    )
    # critic model
    self.critic = GPTModelCritic(
      num_tokentypes=num_tokentypes,
      parallel_output=parallel_output,
      pre_process=pre_process,
      post_process=post_process,
      prefix_lm=prefix_lm,
      args=critic_args
    )
    
    # running moments
    # print_with_rank("> initialize running moments ...")
    self.running = RunningMoments()
    print_with_rank("> running moments are initialized!")
    
  # generate batch of samples  
  def _generate_batch_with_pipeline(
    self,
    prompts:list,
    max_new_tokens,
    tokenizer,
    eos_token_id,
    max_seq_len,
    min_new_tokens=None,
    pipeline_batch_size=None,
    micro_batch_size=1,
    greedy=False,
    temperature=1,
    top_k=0,
    top_p=1,
    ):
    return generate_samples_with_pipeline(
      self,
      prompts,
      max_new_tokens,
      tokenizer,
      eos_token_id,
      max_seq_len,
      min_new_tokens=min_new_tokens,
      pipeline_batch_size=pipeline_batch_size,
      micro_batch_size=micro_batch_size,
      greedy=greedy,
      temperature=temperature,
      top_k=top_k,
      top_p=top_p,
    )

  def _generate_batched(
    self,
    prompts:list,
    max_new_tokens,
    tokenizer,
    eos_token_id,
    max_seq_len,
    min_new_tokens=None,
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
      'generated_texts': []  ,
      "prompts": [] 

    }
    batch_starts = list(range(0,len(prompts),batch_size))
    if dist.get_rank() == 0:
      iter = tqdm(batch_starts, total=len(batch_starts))
    else:
      iter = batch_starts

    # generate sample by batch
    for batch_start in iter:
      
      prompts_batch = prompts[batch_start:(batch_start+batch_size)]
      if callable(min_new_tokens):
        min_new_tokens_ = torch.cuda.LongTensor(
          [min_new_tokens() for i in range(len(prompts_batch))]
          )
      else:
        min_new_tokens_ = min_new_tokens

      r = generate_samples_ppo(
        self,
        prompts_batch,
        max_new_tokens,
        tokenizer,
        eos_token_id,
        max_seq_len,
        min_new_tokens=min_new_tokens_,
        recompute=recompute,
        greedy=greedy,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
      )

      if r is None:
        ret = r 
      else:
          for k,v in ret.items():  
            v.extend(r[k])
    return ret
  
  def compute_reward_with_kl_divergence(
      self,
      prompts,
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
    kls_token = []

    for prompt,score,ref_logprob,logprob in zip(
      prompts,scores,ref_logprob_list,logprob_list):
      kls.append((logprob - ref_logprob).sum())
      kls_token.append((logprob - ref_logprob).mean())
      non_score_rewards = -kl_ctl.value * (logprob - ref_logprob
      )
      non_score_rewards_list.append(non_score_rewards)
      rewards = non_score_rewards.clone()

      # clip reward score
      score_clip = torch.clamp(torch.tensor(score,device=logprob.device), -self.ppo_config['cliprange_reward'],
                                  self.ppo_config['cliprange_reward'])
      try:
        rewards[0][-1] += score_clip  # 最后一个位置的reward加上句子的评分
      except:
        print(f'prompt: {prompt}, rewards: {rewards}, score: {score}, ref_logprob: {ref_logprob}, logprob: {logprob}')
        raise
      rewards_list.append(rewards)

    return rewards_list,non_score_rewards_list,kls,kls_token

  def make_experience(self,prompts,reward_fn,ref_model_fn,kl_ctl):

    ppo_generate_batch_size = getattr(self.args,'ppo_generate_batch_size',self.args.micro_batch_size)
    ppo_generate_microbatches = getattr(self.args,"ppo_generate_microbatches",mpu.get_pipeline_model_parallel_world_size())
    max_new_tokens=self.ppo_config['gen_kwargs']['max_new_tokens']
    min_new_tokens = self.ppo_config['gen_kwargs'].get('min_new_tokens',None)

    if isinstance(min_new_tokens,str):
      assert min_new_tokens == "random", min_new_tokens
      min_new_tokens = lambda : random.choice(list(range(1,50)))

    timing = {}
    t0 = time.time()
    checkpoint_activations = self.args.checkpoint_activations
    self.args.checkpoint_activations = False

    # generate ppo samples
    generated_data = self._generate_batch_with_pipeline(
      prompts,            
      max_new_tokens,     # max new token count
      self.tokenizer,
      self.tokenizer.eod,  #这里设置-1是为了保证每次都产生足量的数据。
      self.args.seq_length,           # max_seq_length
      min_new_tokens=min_new_tokens,  # 最少的生成token数量
      pipeline_batch_size=ppo_generate_microbatches,    # number of ppo generation micro batches
      micro_batch_size=ppo_generate_batch_size,         # ppo generation micro batchsize
      greedy=self.args.greedy,
      temperature=self.args.temperature,
      top_k=self.args.top_k,
      top_p=self.args.top_p,
    )
    torch.cuda.empty_cache()
    self.args.checkpoint_activations = checkpoint_activations

    print_rank_0(f'sample {len(prompts)}, max new tokens: {max_new_tokens} time: {time.time()-t0}')

    if generated_data is not None:
      for i, generated_text in enumerate(generated_data['generated_texts']):
        if not generated_text:
          print_rank_0(f'{prompts[i]} generate empty')
     
    timing['timing/ppo/sample_time'] = time.time()-t0

    # 最后一个stage的才会进行计算: old_logprob_list, old_values_list
    if generated_data is not None:
      query_tensors_list = generated_data['query_tensors']
      response_tensors_list = generated_data['response_tensors']
      # 这里需要注意cpu不支持half的softmax
      old_logprob_list = [
        self.logprobs_of_labels(
          lm_logit.float(),
          response_tensor
        ).half()
        for lm_logit, response_tensor in zip(generated_data['lm_logits'],response_tensors_list)
        ]
      old_values_list = generated_data['values']
      old_generated_text_list = generated_data['generated_texts']
      generated_texts = generated_data['generated_texts']
      response_tensor_len_list = [response_tensor.shape[1] for response_tensor in response_tensors_list]
    else:
      query_tensors_list = []
      response_tensors_list = []
      old_logprob_list = []
      old_values_list = []
      response_tensor_len_list = []
      generated_texts = []

    # reward scores: old_rewards_list, ref_logprob_list
    kl_mean = None
    kl_token_mean = None
    old_rewards_list = None
    reward_list=[]
    if mpu.is_last_stage_and_scr_rank():
      texts = [prompt + generated_text for prompt,generated_text in zip(prompts,old_generated_text_list)]
      query_py_list = [q.tolist()[0] for q in query_tensors_list]
      response_py_list = [r.tolist()[0] for r in response_tensors_list]
      t1 = time.time()
      ref_logprob_list = ref_model_fn({
        'query_tensors':query_py_list,'response_tensors':response_py_list
        })
      t2 = time.time()
      old_rewards_list = reward_fn(texts)
      t3 = time.time()

      timing['timing/ppo/ref_call_time'] = t2-t1
      timing['timing/ppo/reward_call_time'] = t3 -t2

      # 对reward值进行归一化
      if self.ppo_config['reward_normalize']:
        print('execute reward_normalize')
        # old_rewards_tensor = torch.cat([i.squeeze(0) for i in old_rewards_list])
        # old_rewards_tensor_mean, old_rewards_tensor_std = self.running.update(old_rewards_tensor)
        # # old_rewards_tensor_mean = old_rewards_tensor.mean()
        # # old_rewards_tensor_std = old_rewards_tensor.std()
        # old_rewards_list = [
        #   (old_rewards - old_rewards_tensor_mean) / old_rewards_tensor_std + old_rewards_tensor_mean
        #   for old_rewards in old_rewards_list
        #   ]
        # print(f">> running reward mean: {old_rewards_tensor_mean.item()}, reward std: {old_rewards_tensor_std.item()}")
        old_rewards_mean, old_rewards_std = self.running.update(old_rewards_list)
        old_rewards_list = [
          (old_rewards - old_rewards_mean) / old_rewards_std + old_rewards_mean
          for old_rewards in old_rewards_list
          ]
        print(f">> running reward mean: {old_rewards_mean}, reward std: {old_rewards_std}")

      reward_list = copy.deepcopy(old_rewards_list)
      # combined ppo objective: reward + kl_divergence
      old_rewards_list,non_score_rewards_list,kls,kls_token = self.compute_reward_with_kl_divergence(
        prompts,
        old_rewards_list,
        ref_logprob_list,
        old_logprob_list,
        kl_ctl
      )

      kl_mean = sum(kls)/len(kls)
      kl_token_mean = sum(kls_token)/len(kls_token)
      for i,prompt in enumerate(prompts[:3]):
        print(
          f'>> 正在采样: {i}, {prompt}->{generated_data["generated_texts"][i]},' \
          f'score: {reward_list[i]}, kl: {kls[i]}/{kls_token[i]}, query_len: {query_tensors_list[i].shape[1]}, res_len: {response_tensors_list[i].shape[1]}'
          )
    
    dist.barrier()
    
    return {
      'query_tensors_list':query_tensors_list,
      'response_tensors_list':response_tensors_list,
      'old_logprob_list':old_logprob_list,
      'old_values_list':old_values_list,
      'old_rewards_list':old_rewards_list,
      'reward_list':reward_list,
      'response_tensor_len_list':response_tensor_len_list,
      'timing':timing,
      'kl_mean':kl_mean,
      'kl_token_mean': kl_token_mean,
      'generated_texts': generated_texts
    }
    
  
  def eval_prompts(self,prompts:list):
    """
    给定prompts,返回采样结果
    """
    
    ppo_generate_batch_size = getattr(self.args,'ppo_generate_batch_size',self.args.micro_batch_size)
    # 
    ppo_generate_microbatches = getattr(self.args,"ppo_generate_microbatches",mpu.get_pipeline_model_parallel_world_size())
    max_new_tokens=self.ppo_config['gen_kwargs']['max_new_tokens']
    min_new_tokens = self.ppo_config['gen_kwargs'].get('min_new_tokens',None)
    if isinstance(min_new_tokens,str):
      assert min_new_tokens == "random", min_new_tokens
      min_new_tokens = lambda : random.choice(list(range(1,50)))
    
    checkpoint_activations = self.args.checkpoint_activations
    self.args.checkpoint_activations = False
    t0 = time.time()
    generated_data = self._generate_batch_with_pipeline(
      prompts,
      max_new_tokens,
      self.tokenizer,
      self.tokenizer.eod, #这里设置-1是为了保证每次都产生足量的数据。
      self.args.seq_length,
      min_new_tokens=min_new_tokens, # 最少的生成token数量
      pipeline_batch_size=ppo_generate_microbatches,
      micro_batch_size=ppo_generate_batch_size,
      greedy=self.args.greedy,
      temperature=self.args.temperature,
      top_k=self.args.top_k,
      top_p=self.args.top_p,
    )

    torch.cuda.empty_cache()
    self.args.checkpoint_activations = checkpoint_activations

    print_rank_0(f'eval sample {len(prompts)}, max new tokens: {max_new_tokens} time: {time.time()-t0}')

    if generated_data is not None: #
      generated_texts = generated_data['generated_texts']
    else:
      generated_texts = []

    return {
      'generated_texts': generated_texts
    }


  def get_global_statistics(self, xs: torch.Tensor):
    """
    Computes element-wise mean and variance of the tensor across processes
    """
    from megatron import print_with_rank
    sum_and_count = torch.tensor([xs.sum(), xs.numel()], device=xs.device)
    # print_with_rank('sum_and_count:',sum_and_count)  
    # 根据数据并行group进行全局统计
    dist.all_reduce(sum_and_count,
                    dist.ReduceOp.SUM,
                    group=mpu.get_data_parallel_group())
    # print_with_rank('sum_and_count reduced:',sum_and_count)  
    global_sum, count = sum_and_count
    global_mean = global_sum / count

    sum_var = torch.sum((xs - global_mean)**2)
    dist.all_reduce(
      sum_var, 
      dist.ReduceOp.SUM,
      group=mpu.get_data_parallel_group()
      )
    global_var = sum_var / count
    return global_mean, global_var, count

  def whiten(self,
             xs: torch.Tensor,
             shift_mean=True,
             distributed=True) -> torch.Tensor:
    """Whitens values"""
    if distributed and dist.is_initialized():
      mean, var, _ = self.get_global_statistics(xs)
    else:
      var, mean = torch.var_mean(xs)

    whitened = (xs - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
      whitened += mean
    return whitened

  def get_advantages_and_returns(
      self,
      values,
      rewards,
      response_length,
      use_whitening=True,
  ):
    """
    计算当前advantages与returns
    Args:
        old_values (_type_): _description_
        old_rewards (_type_): _description_
        response_length (_type_): _description_
        """
    lastgaelam = 0
    advantages_reversed = []
  
    for t in reversed(range(response_length)):
      nextvalues = values[:, t + 1] if t < response_length - 1 else 0.0
      delta = rewards[:, t] + self.ppo_config['gamma'] * nextvalues - values[:, t]
      lastgaelam = delta + self.ppo_config['gamma']* self.ppo_config['lam'] * lastgaelam
      advantages_reversed.append(lastgaelam)
    advantages = torch.stack(advantages_reversed[::-1], dim=1)
    # print('origin rewards',rewards.shape,rewards)
    # print('values',values.shape,values)
    # print('origin advantages',advantages.shape,advantages)
    # print(sfg)
    returns = advantages + values
    if use_whitening:
      advantages = self.whiten(advantages)
    # print('whitening advantages',advantages.shape,advantages)
    return advantages.detach(), returns
  
  @staticmethod
  def logprobs_of_labels(logits, labels):
    """Log probabilities of the labels

    These are calculated from the logits.
    """
    # print(f'logits: {logits.shape}, labels: {labels.shape}')
    logprobs = F.log_softmax(logits, dim=-1)
    logprobs_labels = torch.gather(logprobs,
                                   dim=-1,
                                   index=labels.unsqueeze(-1))
    return logprobs_labels.squeeze(-1)

  def get_tensor_stats(self, xs: torch.Tensor, mask: torch.Tensor, n: int):
    """

    Args:
        xs (torch.Tensor): _description_
        mask (torch.Tensor): _description_
        n (int): _description_

    Returns:
        _type_: _description_
    """
    mean = (xs * mask).sum() / n
    minimum = torch.tensor(torch.finfo(xs.dtype).max).cuda().to(xs.dtype)
  
    return dict(
        mean=mean,
        min=torch.where(mask.bool(), xs, minimum).min(),
        max=torch.where(mask.bool(), xs, -minimum).max(),
        std=torch.sqrt(((xs - mean) * mask).pow(2).sum() / n),
    )

  def flatten_dict(
      self,
      d,
      parent_key: str = "",
      sep: str = "/",
  ) -> dict:
    # From: https://stackoverflow.com/a/6027615
    items = []
    for k, v in d.items():
      new_key = parent_key + sep + k if parent_key else k
      if isinstance(v, MutableMapping):
        items.extend(self.flatten_dict(v, new_key, sep=sep).items())
      else:
        items.append((new_key, v))
    return dict(items)

  @staticmethod
  def clip_by_value(x, tensor_min, tensor_max):
    """
    Tensor extenstion to torch.clamp
    https://github.com/pytorch/pytorch/issues/2793#issuecomment-428784713
    """
    clipped = torch.max(torch.min(x, tensor_max), tensor_min)
    return clipped

  def _loss(
      self,
      logprobs,
      values,
      old_logprobs,
      old_values,
      advantages,
      returns,
      mask,
      input_tensor_list=None
  ):
    """PPO objective function.
        References:
        - https://stable-baselines.readthedocs.io/en/master/modules/ppo2.html
        """
    from megatron import print_with_rank
    
    values_clipped = self.clip_by_value(
        values,
        old_values - self.ppo_config['cliprange_value'],
        old_values + self.ppo_config['cliprange_value'],
    )
    n = mask.sum()
    vf_loss1 = (values - returns)**2
    vf_loss2 = (values_clipped - returns)**2
    vf_loss = 0.5 * torch.sum(torch.max(vf_loss1, vf_loss2) * mask) / n
    vf_clipfrac = torch.sum((vf_loss2 > vf_loss1).float() * mask) / n
    
    log_ratio = (logprobs - old_logprobs) * mask
    ratio = torch.exp(log_ratio)
    # Unbiased KL-div estimates (`k3`). Ref: http://joschu.net/blog/kl-approx.html
    with torch.no_grad():
      approx_kl = torch.sum((ratio - 1) - log_ratio) / n

    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * torch.clamp(
        ratio,
        1.0 - self.ppo_config['cliprange'],
        1.0 + self.ppo_config['cliprange'],
    )
    pg_loss = torch.sum(torch.max(pg_loss1, pg_loss2) * mask) / n
    pg_clipfrac = torch.sum((pg_loss2 > pg_loss1).float() * mask) / n
  

    # ATT 仅仅作为调试
    loss = pg_loss + self.ppo_config['vf_coef'] * vf_loss
    
    mean_entropy = (-logprobs * mask).sum(axis=-1).mean().item()

    stats = dict(
        losses=dict(
            total_loss=loss.item(),
            policy_loss=pg_loss.item(),
            value_loss=vf_loss.item(),
        ),
        values=dict(
            self.get_tensor_stats(values, mask, n),
            values_error=torch.sum(((values - returns) * mask)**2) / n,
            clipfrac=vf_clipfrac,
        ),
        old_values=self.get_tensor_stats(old_values, mask, n),
        returns=self.get_tensor_stats(returns, mask, n),
        policy=dict(approx_kl=approx_kl.item(), clipfrac=pg_clipfrac.item(),mean_entropy=mean_entropy),
        ratio=(ratio * mask).sum() / n,
        padding_percentage=n / mask.numel(),
    )

    return loss, self.flatten_dict(stats)

  def loss(
    self, 
    old_logprobs,
    old_values,
    old_rewards,
    response_mask,
    query_tensors_len,
    input_ids,
    logits, 
    values
  ):
    """_summary_

      Args:
          old_logprobs (_type_):  [batch,max_response_len]
          old_values (_type_):  [batch,max_response_len]
          old_rewards (_type_):  [batch,max_response_len]
          input_ids: [batch,max_seq_len]
          logits: : [batch,max_seq_len]
          values: [batch,max_seq_len]
    """
    from megatron import print_with_rank

    batch_size,max_response_length = old_logprobs.shape
    assert batch_size == logits.shape[0]
    assert max_response_length <= input_ids.shape[1]

    # response_lengths = [old_logprobs_list[i].shape[1] for i in range(batch_size)]
    # max_response_length = max(response_lengths)
    # 计算advantage 与 return
    old_rewards = old_rewards * response_mask
    old_values = old_values * response_mask
   
    advantages, returns = self.get_advantages_and_returns(
        old_values, old_rewards, max_response_length)
  
    values_pred = values
    
    # TODO 使用向量运算
    logprobs_list = []
    values_pred_list = []
    input_tensor_list = []
    for i in range(batch_size):
      start = query_tensors_len[i] -1
      end = start + max_response_length
      # logprobs_list.append(logprobs[i, start:end])
      response_tensor = input_ids[i,start+1:start+1+max_response_length]
      logprobs_list.append(self.logprobs_of_labels(logits[i,start:end,:],response_tensor))
      # logprobs_list.append(self.logprobs_of_labels(logits[i,start:end,:],response_tensor))
      values_pred_list.append(values_pred[i, start:end])
      input_tensor_list.append(input_ids[i,:query_tensors_len[i]+max_response_length])
    # [batch,max_response_len]
    logprobs = torch.stack(logprobs_list,dim=0)
    values_pred = torch.stack(values_pred_list,dim=0)

    # print('logprobs',logprobs[0])

    # print('old_logprobs',old_logprobs[0])

    # print(sdg)
      
    loss, loss_stat = self._loss(logprobs, values_pred, old_logprobs,
                                 old_values, advantages, returns, response_mask,
                                 input_tensor_list=input_tensor_list
                                 )

    # print_with_rank('loss shape:',loss)       
    return loss, loss_stat

  def set_input_tensor(self, input_tensor):
    """
    See megatron.model.transformer.set_input_tensor()
    这需要区分 actor/critic
    """
    if input_tensor is not None:
      self.actor.language_model.set_input_tensor(
        input_tensor[:,:,:self.actor_args.hidden_size]
        )
      
      self.critic.language_model.set_input_tensor(
        input_tensor[:,:,self.actor_args.hidden_size:]
      )
    else:
      self.actor.language_model.set_input_tensor(None)
      self.critic.language_model.set_input_tensor(None)

  
  def forward(
      self,
      input_ids,
      position_ids,
      attention_mask,
      old_logprobs=None,
      old_values=None,
      old_rewards=None,
      response_mask=None,
      query_tensors_len=None,
      # labels=None,
      tokentype_ids=None,
      layer_past=None,
      get_key_value=False,
      forward_method_parallel_output=None
  ):
    """_summary_

        Args:
            input_ids (_type_): [batch,max_seq_len]
            position_ids (_type_): [batch,max_seq_len]
            attention_mask (_type_): [batch,max_seq_len]
            
            response_tensors : [batch,max_response_len].
            old_logprobs : [batch,max_response_len].
            old_values : [batch,max_response_len].
            old_rewards :[batch,max_response_len].
            response_mask: [batch,max_response_len]
            query_tensors_len: [batch]
            labels : _description_. Defaults to None.
            tokentype_ids (_type_, optional): _description_. Defaults to None.
            layer_past (_type_, optional): _description_. Defaults to None.
            get_key_value (bool, optional): _description_. Defaults to False.
            forward_method_parallel_output (_type_, optional): _description_. Defaults to None.
            curriculum_seqlen (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """

    # [batch, seq, hidden_size]
    from megatron import print_with_rank
    # if not self.pre_process:
      # 需要将hidden states 解包
    
    if get_key_value is not None and layer_past is not None:
      actor_layer_past,critic_layer_past = layer_past
    else:
      actor_layer_past,critic_layer_past = layer_past,layer_past

    # 计算actor logits
    # [batch,maxlen,actor_hidden_size]
    actor_lm_output = self.actor.language_model(input_ids,
                                    position_ids,
                                    attention_mask,
                                    layer_past=actor_layer_past,
                                    get_key_value=get_key_value)
    #[batch,maxlen,critic_hidden_size]
    critic_lm_output = self.critic.language_model(input_ids,
                                    position_ids,
                                    attention_mask,
                                    layer_past=critic_layer_past,
                                    get_key_value=get_key_value
                                    )
    if get_key_value:
      actor_lm_output,actor_presents = actor_lm_output
      critic_lm_output,critic_presents = critic_lm_output
    # print_with_rank(f'>> input_ids: {input_ids.size()}, critic_lm_output: {actor_lm_output.size()}, '\
    #                 f'post_process: {self.post_process}, attention_mask:{attention_mask.size()}')
    if self.post_process:
      # 计算lm loss
      # [batch,seq,vocab_size]
      # 暂时不进行并行输出，尽管这样不利于显存占用
      actor_lm_output_logits = post_language_model_processing(
        actor_lm_output, 
        None,
        self.actor.word_embeddings_weight(),
        False,
        False,
        forward_method_parallel_output,
        self.actor.fp16_lm_cross_entropy
        )
      
      v_head_output = self.critic.v_head(critic_lm_output)
      v_head_output = v_head_output.squeeze(-1)

      # 预测阶段直接返回lm_logits
      if old_logprobs is None:
        if get_key_value:
          return (actor_lm_output_logits,v_head_output),(actor_presents,critic_presents)
        return actor_lm_output_logits,v_head_output
 
      loss, loss_stat = self.loss(
        old_logprobs,
        old_values, 
        old_rewards, 
        response_mask,
        query_tensors_len,
        input_ids,
        actor_lm_output_logits,
        v_head_output
        )
      return loss, loss_stat
    else:
      # print_with_rank(f'get_key_value: {get_key_value}, actor_lm_output:{actor_lm_output.size()}, critic_lm_output: {critic_lm_output.size()}')
      # 为了方便通信，将actor和critic两个网络的输出拼接起来
      hidden_state = torch.cat((actor_lm_output,critic_lm_output),dim=-1)
      if get_key_value:
        return hidden_state,(actor_presents,critic_presents)
      return hidden_state
        

  def state_dict_for_save_checkpoint(self,
                                     destination=None,
                                     prefix='',
                                     keep_vars=False):
   
    # 模型保存，分别保存actor以及critic的参数
    state_dict_ = {}

    state_dict_['actor'] = self.actor.state_dict_for_save_checkpoint(
      destination=destination,
      prefix = prefix,
      keep_vars=keep_vars
    )
    state_dict_['critic'] = self.critic.state_dict_for_save_checkpoint(
      destination=destination,
      prefix = prefix,
      keep_vars=keep_vars
    )
    return state_dict_
  
  def load_state_dict(self, state_dict, strict=True):
    """
    分别加载actor与critic
    Args:
        state_dict (_type_): _description_
        strict (bool, optional): _description_. Defaults to True.
    """
    self.actor.load_state_dict(
      state_dict['actor'],
      strict=strict
    )
    self.critic.load_state_dict(
      state_dict['critic'],
      strict=strict
    )

  

class GPTModelWithPPOValueHead(GPTModel):
  """
  在GPTModel基础上增加值函数head,
  这里的实现中值函数和policy没有分离
  """
  default_ppo_config = dict(
      num_rollouts=128,
      # chunk_size=128,
      ppo_epochs=4,
      init_kl_coef=0.2,
      target=6,
      horizon=10000,
      gamma=1,
      lam=0.95,
      cliprange=0.2,
      cliprange_value=0.2,
      vf_coef=0.1,
      scale_reward="ignored",
      ref_mean=None,
      adap_kl_ctrl=True,
      ref_std=None,
      cliprange_reward=0.2,
      gen_kwargs=dict(max_new_tokens=40,
                      top_k=0,
                      top_p=1.0,
                      do_sample=True,
                      temperature=1),
  )

  def __init__(
      self,
      num_tokentypes=0,
      parallel_output=True,
      pre_process=True,
      post_process=True,
      args=None
      ):
    if args is None:
      args = get_args()
    super().__init__(num_tokentypes=num_tokentypes,
                     parallel_output=parallel_output,
                     pre_process=pre_process,
                     post_process=post_process,
                     prefix_lm=False,
                     args=args
                     )
    # args = get_args()
    # 读取 ppo config
    with open(args.ppo_config_file) as f:
      ppo_config = json.load(f)
      self.ppo_config = self.default_ppo_config.copy()
      self.ppo_config.update(**ppo_config)
    

    if self.post_process:
      # Layernorm on the input data.
      # self.input_layernorm = LayerNorm(args.hidden_size,
      #                                eps=args.layernorm_epsilon)
      
      # self.input_layernorm = LayerNorm(args.hidden_size,
      #                                eps=args.layernorm_epsilon)
      self.v_head = self.initial_value_head(args.hidden_size,args=args)
      # self.v_head_dropout = 
      # self.v_head = nn.Sequential(
      #   nn.Dropout(0.1),
      #   # LayerNorm(args.hidden_size, eps=args.layernorm_epsilon),
      #   nn.Linear(args.hidden_size, 1)
      # )

      # self.v_head.weight.data.normal_(mean=0.0, std=0.2)
       
      # self.v_head = nn.Sequential(
      #   nn.LayerNorm(args.hidden_size),
      #   nn.Linear(args.hidden_size,args.hidden_size*2),
      #   nn.ReLU(),
      #   nn.Linear(args.hidden_size*2,1),

      # ) 

      # self.initial_value_head(args.hidden_size)

  def initial_value_head(self, hidden_size,args=None):
    v_head = ParallelLinear(
        hidden_size,
        1,
        bias=True,
        input_is_parallel=False,
        args=args
    )
    return v_head

  def get_global_statistics(self, xs: torch.Tensor):
    """
        Computes element-wise mean and variance of the tensor across processes
        """
    from megatron import print_with_rank
    sum_and_count = torch.tensor([xs.sum(), xs.numel()], device=xs.device)
    # print_with_rank('sum_and_count:',sum_and_count)  
    # 根据数据并行group进行全局统计
    dist.all_reduce(sum_and_count,
                    dist.ReduceOp.SUM,
                    group=mpu.get_data_parallel_group())
    # print_with_rank('sum_and_count reduced:',sum_and_count)  
    global_sum, count = sum_and_count
    global_mean = global_sum / count

    sum_var = torch.sum((xs - global_mean)**2)
    dist.all_reduce(
      sum_var, 
      dist.ReduceOp.SUM,
      group=mpu.get_data_parallel_group()
      )
    global_var = sum_var / count
    return global_mean, global_var, count

  def whiten(self,
             xs: torch.Tensor,
             shift_mean=True,
             distributed=True) -> torch.Tensor:
    """Whitens values"""
    if distributed and dist.is_initialized():
      mean, var, _ = self.get_global_statistics(xs)
    else:
      var, mean = torch.var_mean(xs)

    whitened = (xs - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
      whitened += mean
    return whitened

  def get_advantages_and_returns(
      self,
      values,
      rewards,
      response_length,
      use_whitening=True,
  ):
    """
        计算当前advantages与returns
        Args:
            old_values (_type_): _description_
            old_rewards (_type_): _description_
            response_length (_type_): _description_
        """
    lastgaelam = 0
    advantages_reversed = []
    # print('values',values)
    # print('rewards',rewards)
    for t in reversed(range(response_length)):
      nextvalues = values[:, t + 1] if t < response_length - 1 else 0.0
      delta = rewards[:, t] + self.ppo_config['gamma'] * nextvalues - values[:, t]
      lastgaelam = delta + self.ppo_config['gamma']* self.ppo_config['lam'] * lastgaelam
      advantages_reversed.append(lastgaelam)
    advantages = torch.stack(advantages_reversed[::-1], dim=1)
    # print('advantages',advantages.shape)
    returns = advantages + values
    if use_whitening:
      advantages = self.whiten(advantages)
    
    # print('advantages after',advantages)
    # print('returns after',returns)
    return advantages.detach(), returns
  
  @staticmethod
  def logprobs_of_labels(logits, labels):
    """Log probabilities of the labels

    These are calculated from the logits.
    """
    # print(f'logits: {logits.shape}, labels: {labels.shape}')
    logprobs = F.log_softmax(logits, dim=-1)
    logprobs_labels = torch.gather(logprobs,
                                   dim=-1,
                                   index=labels.unsqueeze(-1))
    return logprobs_labels.squeeze(-1)

  def get_tensor_stats(self, xs: torch.Tensor, mask: torch.Tensor, n: int):
    """

        Args:
            xs (torch.Tensor): _description_
            mask (torch.Tensor): _description_
            n (int): _description_

        Returns:
            _type_: _description_
        """
    mean = (xs * mask).sum() / n
    minimum = torch.tensor(torch.finfo(xs.dtype).max).cuda().to(xs.dtype)
    # print(xs,minimum)
    # torch.where(mask.bool(), xs, minimum)
    # print(mask)
    # print(xs)
    return dict(
        mean=mean,
        min=torch.where(mask.bool(), xs, minimum).min(),
        max=torch.where(mask.bool(), xs, -minimum).max(),
        std=torch.sqrt(((xs - mean) * mask).pow(2).sum() / n),
    )

  def flatten_dict(
      self,
      d,
      parent_key: str = "",
      sep: str = "/",
  ) -> dict:
    # From: https://stackoverflow.com/a/6027615
    items = []
    for k, v in d.items():
      new_key = parent_key + sep + k if parent_key else k
      if isinstance(v, MutableMapping):
        items.extend(self.flatten_dict(v, new_key, sep=sep).items())
      else:
        items.append((new_key, v))
    return dict(items)

  @staticmethod
  def clip_by_value(x, tensor_min, tensor_max):
    """
    Tensor extenstion to torch.clamp
    https://github.com/pytorch/pytorch/issues/2793#issuecomment-428784713
    """
    clipped = torch.max(torch.min(x, tensor_max), tensor_min)
    return clipped

  def _loss(
      self,
      logprobs,
      values,
      old_logprobs,
      old_values,
      advantages,
      returns,
      mask,
      input_tensor_list=None
  ):
    """PPO objective function.
        References:
        - https://stable-baselines.readthedocs.io/en/master/modules/ppo2.html
        """
    from megatron import print_with_rank
    
    values_clipped = self.clip_by_value(
        values,
        old_values - self.ppo_config['cliprange_value'],
        old_values + self.ppo_config['cliprange_value'],
    )
    n = mask.sum()

    # print('valid num:', n)

    # print('values,returns', values,returns)
    # print(sfg)

    # print_with_rank('mask sum:',n)   
    vf_loss1 = (values - returns)**2
    vf_loss2 = (values_clipped - returns)**2
    vf_loss = 0.5 * torch.sum(torch.max(vf_loss1, vf_loss2) * mask) / n
    vf_clipfrac = torch.sum((vf_loss2 > vf_loss1).float() * mask) / n

    # print('vf_loss',vf_loss)
    # print('advantages',advantages)
    
    
    log_ratio = (logprobs - old_logprobs) * mask
    ratio = torch.exp(log_ratio)
    # Unbiased KL-div estimates (`k3`). Ref: http://joschu.net/blog/kl-approx.html
    with torch.no_grad():
      approx_kl = torch.mean((ratio - 1) - log_ratio)

    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * torch.clamp(
        ratio,
        1.0 - self.ppo_config['cliprange'],
        1.0 + self.ppo_config['cliprange'],
    )
    pg_loss = torch.sum(torch.max(pg_loss1, pg_loss2) * mask) / n
    pg_clipfrac = torch.sum((pg_loss2 > pg_loss1).float() * mask) / n
    

    # ATT 仅仅作为调试
    loss = pg_loss + self.ppo_config['vf_coef'] * vf_loss
    # loss = 0.0*pg_loss + 0.0 * vf_loss
    
    # print('ratio',ratio)
    # print('pg_loss1',pg_loss1)
    # print('pg_loss2',pg_loss2)
    # print('pg_loss',pg_loss)
    # print('vf_loss',vf_loss)
    # print('approx_kl',approx_kl)
    # print('logprobs vs old_logprobs',old_logprobs[:,:5],logprobs[:,:5])
    # print('values vs old_values',values[:,:5],old_values[:,:5])
    # tokenizer = get_tokenizer()
    
    # print([tokenizer.detokenize(input_ids.tolist()) for input_ids in input_tensor_list])
    # print(input_tensor_list)
    # print('mask num',mask.sum())
    
   
    # grad = torch.autograd.grad(outputs=loss,inputs=self.v_head[2].weight)
    # print('sdfgdsfhgdfh,loss:',loss)
    # print('weight grad',self.v_head[2].weight.grad)
    # print(sdrg)
    # mean_kl = ((logprobs- old_logprobs)) * mask.sum(axis=-1).mean()
    # mean_kl = kl_list.mean()
    mean_entropy = (-logprobs * mask).sum(axis=-1).mean().item()

    stats = dict(
        losses=dict(
            total_loss=loss.item(),
            policy_loss=pg_loss.item(),
            value_loss=vf_loss.item(),
        ),
        values=dict(
            self.get_tensor_stats(values, mask, n),
            values_error=torch.sum(((values - returns) * mask)**2) / n,
            clipfrac=vf_clipfrac,
        ),
        old_values=self.get_tensor_stats(old_values, mask, n),
        returns=self.get_tensor_stats(returns, mask, n),
        policy=dict(approx_kl=approx_kl.item(), clipfrac=pg_clipfrac.item(),mean_entropy=mean_entropy),
        ratio=(ratio * mask).sum() / n,
        padding_percentage=n / mask.numel(),
    )

    return loss, self.flatten_dict(stats)

  def loss(self, 
    old_logprobs,
    old_values,
    old_rewards,
    response_mask,
    query_tensors_len,
    input_ids,
    logits, 
    values
  ):
    """_summary_

        Args:
            old_logprobs (_type_):  [batch,max_response_len]
            old_values (_type_):  [batch,max_response_len]
            old_rewards (_type_):  [batch,max_response_len]
            input_ids: [batch,max_seq_len]
            logits: : [batch,max_seq_len]
            values: [batch,max_seq_len]
        """
    from megatron import print_with_rank

    batch_size,max_response_length = old_logprobs.shape
    assert batch_size == logits.shape[0]
    assert max_response_length <= input_ids.shape[1]

    # response_lengths = [old_logprobs_list[i].shape[1] for i in range(batch_size)]
    # max_response_length = max(response_lengths)
    # 计算advantage 与 return
    old_rewards = old_rewards * response_mask
    old_values = old_values * response_mask
   
    advantages, returns = self.get_advantages_and_returns(
        old_values, old_rewards, max_response_length)
    
    
    # print('max_response_length',max_response_length)
    # tokens = torch.cat((query_tensors, response_tensors), dim=1)
    # print_with_rank('tokens shape:',tokens.shape)
    # tokenizer = get_tokenizer()
    # attention_mask = torch.ones_like(tokens).long().to(tokens.device)
    # outputs = self.model(tokens, attention_mask, return_dict=True)
    # logits = outputs.logits
    values_pred = values
    # values_pred = values_pred[:, :-1]
    # 取出固定位置的logprob做近似 kl
    # logprobs = self.logprobs_of_labels(logits[:, :-1, :], input_ids[:, 1:])
    # print('-----------------------------------')
    # print('logits',logits[:, 5, :].max(),logits[:, 5, :].min())
    # tokenizer = get_tokenizer()
    
    # print('sgsg',old_values.shape,old_logprobs.shape,values_pred.shape)
    # print(sgsg)
    # TODO 使用向量运算
    logprobs_list = []
    values_pred_list = []
    input_tensor_list = []
    for i in range(batch_size):
      start = query_tensors_len[i] -1
      end = start + max_response_length
      # logprobs_list.append(logprobs[i, start:end])
      response_tensor = input_ids[i,start+1:start+1+max_response_length]
      # print('label to gather logits',response_tensor,tokenizer.detokenize(response_tensor.tolist()))
      # logits_argmax = logits[i,start:end,:].argmax(dim=-1)
      # logits_argmax_res = logits[i,start+1:end+1,:].argmax(dim=-1)
      # from torch.distributions import Categorical

      # print('logits_argmax:',logits_argmax,', entropy:',Categorical(logits=logits[i,start:end,:]).entropy(),', logits max text: ',tokenizer.detokenize(logits_argmax.tolist()))
      # print('logits_argmax response location',logits_argmax_res,tokenizer.detokenize(logits_argmax_res.tolist()))
      logprobs_list.append(self.logprobs_of_labels(logits[i,start:end,:],response_tensor))
      # logprobs_list.append(self.logprobs_of_labels(logits[i,start:end,:],response_tensor))
      values_pred_list.append(values_pred[i, start:end])
      input_tensor_list.append(input_ids[i,:query_tensors_len[i]+max_response_length])
    # [batch,max_response_len]
    logprobs = torch.stack(logprobs_list,dim=0)
    values_pred = torch.stack(values_pred_list,dim=0)
    # print('advantage',advantages)
    # print('returns',returns)

    # print('query_tensors_len',query_tensors_len)
    # print('logprobs shape',logprobs.shape)

    # print(
    #   'old_values',
    #   old_values.max(),
    #   old_values.min(),
    #   old_values.abs().mean(),
    #   old_values.shape,
    #   (old_values >0).sum(),
    #   )
    # print('old_values',old_values[0,:10])

    # print(
    #   'values_pred',
    #   values_pred.max(),
    #   values_pred.min(),
    #   values_pred.abs().mean(),
    #   values_pred.shape,
    #   (values_pred >0).sum(),
    #   )
    # print('values_pred',values_pred[0,:10])
    # print('returns',returns)

    # print('logprobs',logprobs)
    # print('old_logprobs',old_logprobs)
    # print('old_rewardscccc',old_rewards)

    # print(sdg)

    # print_with_rank('logprobs shape:',logprobs.shape) 
      
    loss, loss_stat = self._loss(logprobs, values_pred, old_logprobs,
                                 old_values, advantages, returns, response_mask,
                                 input_tensor_list=input_tensor_list
                                 )

    # print_with_rank('loss shape:',loss)       
    return loss, loss_stat

  def forward(
      self,
      input_ids,
      position_ids,
      attention_mask,
      old_logprobs=None,
      old_values=None,
      old_rewards=None,
      response_mask=None,
      query_tensors_len=None,
      # labels=None,
      tokentype_ids=None,
      layer_past=None,
      get_key_value=False,
      forward_method_parallel_output=None,
  ):
    """_summary_

        Args:
            input_ids (_type_): [batch,max_seq_len]
            position_ids (_type_): [batch,max_seq_len]
            attention_mask (_type_): [batch,max_seq_len]
            
            response_tensors : [batch,max_response_len].
            old_logprobs : [batch,max_response_len].
            old_values : [batch,max_response_len].
            old_rewards :[batch,max_response_len].
            response_mask: [batch,max_response_len]
            query_tensors_len: [batch]
            labels : _description_. Defaults to None.
            tokentype_ids (_type_, optional): _description_. Defaults to None.
            layer_past (_type_, optional): _description_. Defaults to None.
            get_key_value (bool, optional): _description_. Defaults to False.
            forward_method_parallel_output (_type_, optional): _description_. Defaults to None.
            curriculum_seqlen (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """

    # [batch, seq, hidden_size]
    from megatron import print_with_rank
    # print_with_rank('开始 language_model ')
    lm_output = self.language_model(input_ids,
                                    position_ids,
                                    attention_mask,
                                    layer_past=layer_past,
                                    get_key_value=get_key_value)

    if self.post_process:
      # 计算lm loss
      # [batch,seq,vocab_size]
      # 暂时不进行并行输出，尽管这样不利于显存占用
      lm_logits = post_language_model_processing(
        lm_output, 
        None,
        self.word_embeddings_weight(),
        get_key_value,
        False,
        forward_method_parallel_output,
        self.fp16_lm_cross_entropy
        )

      # print('sfghdfhd',type(lm_logits))

      # 这里值函数的梯度不进行回传
      # print('lm_output.shape',lm_output.shape)
      if get_key_value:
        lm_output,presents = lm_output
      

      # v_head_output = self.input_layernorm(lm_output)
      v_head_output = self.v_head(lm_output)
      # v_head_output = self.input_layernorm(v_head_output)
      v_head_output = v_head_output.squeeze(-1)
    

      # 预测阶段直接返回lm_logits
      if old_logprobs is None:
        if get_key_value:
          return (lm_logits[0],v_head_output),lm_logits[1]
        return lm_logits, v_head_output
 
      
      # print('input_ids shape',input_ids.shape,input_ids[:,:20])
      
      # print('lm_output ',lm_output.shape,lm_output[:,0,:10])
      # print('lm_output',lm_output.max(),lm_output.min(),lm_output.abs().mean())
      # print('logits')
      # print('v_head weights',self.v_head.layer.weight[:5,:5])
      # print('v_head weights',self.v_head.layer.weight[:5,:5])
      # print('v_head weights',lm_output*self.v_head.layer.weight*)
      # print('v_head_output',v_head_output.max(),v_head_output.min(),v_head_output.abs().mean(),v_head_output.shape)
      # print(
      #   'lm_output num',lm_output.numel(),
      #   'lm_output positive num', (lm_output > 0).sum(),
      #    'v_head_output num', v_head_output.numel(),
      #    'v_head_output positive num', (v_head_output>0).sum(),
      #    'v_head wieght positive num', (self.v_head.weight.data >0).sum()
      #    )

      
      
        
      # print('old_values',old_values.shape)

      
      # print(sfg)
      # print('lm_output shape',lm_output.shape)
      # print()
      # print('lm_output after layernorm',output_layernormed)

      # print('v_head_output',v_head_output.max(),v_head_output.min(),v_head_output.abs().mean(),v_head_output.shape)
      # print('old rewards',old_rewards)
      # print(sg)
      # 这里仅仅针对bs = 1
      # assert input_ids.shape[0] == 1
      # response_size = old_logprobs.shape[1]
      # query_size = query_tensors.shape[1]

      # # maxlen = query_tensors.shape[1] + response_size
      # # query_tensors = input_ids[:,:(maxlen-response_size)]
      # response_tensors = input_ids[:,query_size:(query_size+response_size)]
      # print('sfgsfgsfgs',v_head_output.shape,old_values.shape)
      loss, loss_stat = self.loss(
        old_logprobs,
        old_values, 
        old_rewards, 
        response_mask,
        query_tensors_len,
        input_ids,
        lm_logits,
         v_head_output
         )
      return loss, loss_stat
    else:
      return lm_output




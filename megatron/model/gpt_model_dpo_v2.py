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
"""实现dpo模型,其中也是有两个网络, 一个是ref模型,另外一个是policy"""

import torch
import torch.nn.init as init
from torch import nn
from tqdm import tqdm
import random
import numpy as np
import json
import time

from megatron import get_args
from megatron import mpu, print_rank_0
from megatron import get_tokenizer
import torch.distributed as dist
from megatron.enums import AttnMaskType

from torch.nn import functional as F

from megatron.model.fused_layer_norm import MixedFusedLayerNorm as LayerNorm
from megatron.model.module import float16_to_fp32

from .gpt_model import GPTModel, post_language_model_processing


class ParallelLinear(nn.Module):
  """Linear layer parallelized over the longer dimension."""

  def __init__(self,
               in_size: int,
               out_size: int,
               init_method=init.xavier_normal_,
               bias=True,
               gather_output=True,
               input_is_parallel=False,
               args=None):
    super().__init__()

    if in_size < out_size:
      self.layer = mpu.ColumnParallelLinear(in_size,
                                            out_size,
                                            bias=bias,
                                            gather_output=gather_output,
                                            init_method=init_method,
                                            skip_bias_add=False,
                                            args=args)
    else:
      self.layer = mpu.RowParallelLinear(in_size,
                                         out_size,
                                         input_is_parallel=input_is_parallel,
                                         init_method=init_method,
                                         skip_bias_add=False,
                                         bias=bias,
                                         args=args)

  def forward(self, x):

    output, bias = self.layer(x)
    # print('v_head_output',output[:5,:5])
    # print('v_head_bias',bias[:5])
    if bias is not None:
      return output + bias
    return output


class ActorRefModel(torch.nn.Module):
  """
  ppo实现,区别actor, ref方法
  Args:
      torch (_type_): _description_
  """

  def __init__(
      self,
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
        ref_args: ref模型需要的参数
        args (_type_): 通用的参数
    """
    super().__init__()
    self.pre_process = pre_process
    self.post_process = post_process
    self.args = args
    # 对应kl权重
    self.kl_coeff = self.args.kl_coeff

    # # 读取 ppo config

    self.tokenizer = get_tokenizer()

    self.actor = GPTModel(num_tokentypes=num_tokentypes,
                          parallel_output=parallel_output,
                          pre_process=pre_process,
                          post_process=post_process,
                          prefix_lm=prefix_lm,
                          args=args
                          )

  @staticmethod
  def logprobs_of_labels(logits, labels):
    """Log probabilities of the labels

    These are calculated from the logits.
    """
    logprobs = F.log_softmax(logits, dim=-1)
    logprobs_labels = torch.gather(logprobs,
                                   dim=-1,
                                   index=labels.unsqueeze(-1))
    return logprobs_labels.squeeze(-1)

  def loss(
    self,
    input_ids,
    actor_lm_output_logits, 
    ref_logprob_list, 
    prompt_lens,
    response_lens
    ):
    """
    计算reward模型的loss
    Args:
        input_ids (_type_): _description_
        rewards (_type_): _description_
        pad_id (_type_): _description_
    """
  

    # Split the inputs and rewards into two parts, chosen and rejected
    assert len(input_ids.shape) == 2
    # print('input_ids.shape',input_ids.shape)
    bs = input_ids.shape[0] // 2
    # seq_len = input_ids.shape[1]
    # print('input_ids.shape',input_ids.shape,rewards.shape,bs)

    # [batch, maxlen]
    chosen_ids = input_ids[:bs]
    rejected_ids = input_ids[bs:]
    # [batch, maxlen]
    chosen_logits = actor_lm_output_logits[:bs,:,:]
    rejected_logits = actor_lm_output_logits[bs:,:,:]

  
    chosen_ref_logprobs = ref_logprob_list[:bs]
    rejected_ref_logprobs = ref_logprob_list[bs:]

    chosen_lens = response_lens[:bs]
    rejected_lens = response_lens[bs:]

    # Compute pairwise loss. Only backprop on the different tokens before padding
    loss = 0
    c_truncated_rewards = 0
    r_truncated_rewards = 0
    for i in range(bs):
      # print(i)
      prompt_len = prompt_lens[i]
      # 按照最大的长度进行score计算
      # chosen_reject_max_len = max(chosen_lens[i],rejected_lens[i])

      chosen_id = chosen_ids[i][prompt_len:prompt_len + chosen_lens[i]]
      rejected_id = rejected_ids[i][prompt_len:prompt_len + rejected_lens[i]]

      chosen_logit = chosen_logits[i,prompt_len-1:prompt_len-1+chosen_lens[i],:]
      rejected_logit = rejected_logits[i,prompt_len-1:prompt_len-1+rejected_lens[i],:]

      chosen_logprob = self.logprobs_of_labels(chosen_logit,chosen_id)
      reject_logprob = self.logprobs_of_labels(rejected_logit,rejected_id)

      chosen_ref_logprob = chosen_ref_logprobs[i][:chosen_lens[i]]
      rejected_ref_logprob = rejected_ref_logprobs[i][:rejected_lens[i]]


      assert chosen_ref_logprob.shape == chosen_logprob.shape, (chosen_ref_logprob.shape, chosen_logprob.shape)
      assert reject_logprob.shape == rejected_ref_logprob.shape, (reject_logprob.shape, rejected_ref_logprob.shape)

      # 计算 logprob 求和
      # chosen_logprob_mean = chosen_logprob.mean()
      # reject_logprob_mean = reject_logprob.mean()

      # chosen_ref_logprob_mean = chosen_ref_logprob.mean()
      # rejected_ref_logprob_mean = rejected_ref_logprob.mean()

      chosen_logprob_mean = chosen_logprob.sum()
      reject_logprob_mean = reject_logprob.sum()

      chosen_ref_logprob_mean = chosen_ref_logprob.sum()
      rejected_ref_logprob_mean = rejected_ref_logprob.sum()


      # print('chosen_logprob_mean',chosen_logprob_mean,chosen_ref_logprob_mean)
      # print('reject_logprob_mean',reject_logprob_mean,rejected_ref_logprob_mean)
      # print(sdfg)

      
      # print('chosen_ids',chosen_ids[i].tolist())
      # print('rejected_ids',rejected_ids[i].tolist())

      # print('chosen_logits', chosen_logits[i])
      # print('rejected_logit', rejected_logits[i])


      # print('chosen_logprob',chosen_logprob.shape,chosen_logprob)
      # print('chosen_ref_logprobs',chosen_ref_logprobs[i].shape,chosen_ref_logprobs[i])

      # print('reject_logprob',reject_logprob.shape,reject_logprob)
      # print('reject_ref_logprob',rejected_ref_logprobs[i].shape,rejected_ref_logprobs[i])

      # print(sfg)

      # c_truncated_reward = self.kl_coeff * (chosen_logprob - chosen_ref_logprobs[i])
      # r_truncated_reward = self.kl_coeff * (reject_logprob - rejected_ref_logprobs[i])
      # loss += -torch.nn.functional.logsigmoid(c_truncated_reward -
      #                                         r_truncated_reward).mean()
      
      c_truncated_reward = self.kl_coeff * (chosen_logprob_mean - chosen_ref_logprob_mean)
      r_truncated_reward = self.kl_coeff * (reject_logprob_mean - rejected_ref_logprob_mean)
      loss += -torch.nn.functional.logsigmoid(c_truncated_reward -
                                              r_truncated_reward)
      
      c_truncated_rewards += c_truncated_reward
      r_truncated_rewards += r_truncated_reward
    
    loss = loss / bs
    c_truncated_reward = c_truncated_rewards/ bs
    r_truncated_reward = r_truncated_rewards / bs

    # print('loss',loss)
    # chosen_mean_scores = torch.stack(chosen_mean_scores)
    # rejected_mean_scores = torch.stack(rejected_mean_scores)
    return {
      'rm_loss':loss,
      'chosen_logprob_diff': c_truncated_reward.detach(),
      'rejected_logprob_diff': r_truncated_reward.detach()
      }


  def set_input_tensor(self, input_tensor):
    """
    See megatron.model.transformer.set_input_tensor()
    这需要区分 actor/critic
    """
    self.actor.language_model.set_input_tensor(
      input_tensor
    )
    

  def forward(
      self,
      input_ids,
      position_ids,
      attention_mask,
      # labels=None,
      # tokentype_ids=None,
      layer_past=None,
      get_key_value=False,
      forward_method_parallel_output=None,
      ref_logprob_list=None,
      prompt_lens=None,
      response_lens=None
      # curriculum_seqlen=None,
      ):
    """

    Args:
        input_ids (_type_): _description_
        position_ids (_type_): _description_
        attention_mask (_type_): _description_
        layer_past (_type_, optional): _description_. Defaults to None.
        get_key_value (bool, optional): _description_. Defaults to False.
        reward_model_training (bool, optional): _description_. Defaults to False.
        pad_id (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """

    # [batch, seq, hidden_size]
    # from megatron import print_with_rank
    # if not self.pre_process:
    # 需要将hidden states 解包

    # if get_key_value is not None and layer_past is not None:
    #   actor_layer_past = layer_past
    # else:
    #   actor_layer_past layer_past, layer_past
      
    
    actor_layer_past = layer_past
    # 计算actor logits
    # [batch,maxlen,actor_hidden_size]
    actor_lm_output = self.actor.language_model(input_ids,
                                                position_ids,
                                                attention_mask,
                                                layer_past=actor_layer_past,
                                                get_key_value=get_key_value)

    
    if get_key_value:
      actor_lm_output, actor_presents = actor_lm_output
      # ref_lm_output, ref_presents = ref_lm_output

    if self.post_process:
      # 计算lm loss
      # actor 计算
      actor_lm_output_logits = post_language_model_processing(
          actor_lm_output, None, self.actor.word_embeddings_weight(), False,
          False, forward_method_parallel_output,
          self.actor.fp16_lm_cross_entropy)
      
      # print('input_ids',input_ids)
      # print('actor_lm_output_logits',actor_lm_output_logits[0].shape,actor_lm_output_logits[0])
      # print('first word embedding ',self.actor.language_model.embedding.word_embeddings.weight)
      # print('head word embedding ',self.actor.word_embeddings_weight())

      # print(sfg)

      
      # print(xf)
      # with torch.no_grad():
      #   ref_lm_output_logits = post_language_model_processing(
      #       ref_lm_output, None, self.ref.word_embeddings_weight(), False,
      #       False, forward_method_parallel_output,
      #       self.ref.fp16_lm_cross_entropy)
      
      # 计算reward
      # ref_lm_output_logits = ref_lm_output_logits.detach()
      # actor_logprobs = self.logprobs_of_labels(actor_lm_output_logits[:,:-1,:],input_ids[:,1:])
      # ref_logprobs = self.logprobs_of_labels(ref_lm_output_logits,input_ids)
      
      # rewards = (actor_logprobs - ref_logprobs)*self.kl_coeff
      
      loss = self.loss(
          input_ids,
          actor_lm_output_logits,
          ref_logprob_list,
          prompt_lens,
          response_lens
      )
      return loss
    else:
      # 为了方便通信，将actor和ref两个网络的输出拼接起来
      hidden_state = actor_lm_output
      if get_key_value:
        return hidden_state, actor_presents
      return hidden_state

  def state_dict_for_save_checkpoint(self,
                                     destination=None,
                                     prefix='',
                                     keep_vars=False):

    # 模型保存，分别保存actor以及ref的参数
    # state_dict_ = {}

    return self.actor.state_dict_for_save_checkpoint(
        destination=destination, prefix=prefix, keep_vars=keep_vars)
    
  def load_state_dict(self, state_dict, strict=True):
    """
    分别加载actor与ref
    Args:
        state_dict (_type_): _description_
        strict (bool, optional): _description_. Defaults to True.
    """
    self.actor.load_state_dict(state_dict, strict=strict)
    # self.ref.load_state_dict(state_dict['ref'], strict=strict)

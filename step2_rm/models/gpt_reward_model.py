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
"""GPT-2 model."""

import torch
import torch.nn.init as init
from torch import nn

from megatron import get_args
from megatron import mpu
from megatron.enums import AttnMaskType
from megatron.model.module import MegatronModule
from megatron.model.gpt_model import GPTModel
from megatron.model.fused_layer_norm import MixedFusedLayerNorm as LayerNorm
from megatron.model.language_model import get_language_model
from megatron.model.utils import init_method_normal
from megatron.model.utils import scaled_init_method_normal

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
    # print('v_head_output',output[:5,:5])
    # print('v_head_bias',bias[:5])
    if bias is not None:
      return output + bias
    return output


class ValuleHead(torch.nn.Module):
  def __init__(self,args) -> None:
    super().__init__()
    self.args = args
    self.dense = ParallelLinear(
          args.hidden_size,
          1,
          bias=True,
          input_is_parallel=False,
          args=args
    )
    if self.args.use_v_head_layernorm:
      self.layernorm = LayerNorm(self.args.hidden_size, args=self.args)
      print('use v_head layernorm ...')
  
  def forward(self,hidden_states):
    if self.args.use_v_head_layernorm:
      hidden_states = self.layernorm(hidden_states)
  
    return self.dense(hidden_states)



class GPTModelCritic(GPTModel):
  """
  实现一个critic模型或者reward模型

  Args:
      GPTModel (_type_): _description_
  """
  def __init__(
      self,
      num_tokentypes=0,
      parallel_output=True,
      pre_process=True,
      post_process=True,
      prefix_lm=False,
      args=None
  ):

    MegatronModule.__init__(self,share_word_embeddings=False)
    if args is None:
      args = get_args()
    
    self.args = args
    
    # args = get_args()
    self.parallel_output = parallel_output
    self.pre_process = pre_process
    self.post_process = post_process

    self.fp16_lm_cross_entropy = args.fp16_lm_cross_entropy
    self.init_method_std = args.init_method_std
    self.num_layers = args.num_layers

    # self.input_layernorm = torch.nn.LayerNorm(self.args.hidden_size)
    self.language_model, self._language_model_key = get_language_model(
      num_tokentypes=num_tokentypes,
      add_pooler=False,
      # TODO: Change naming of class from GPT to something that encapsulate prefix lm.
      encoder_attn_mask_type=AttnMaskType.prefix
      if prefix_lm else AttnMaskType.causal,
      init_method=init_method_normal(self.init_method_std),
      scaled_init_method=scaled_init_method_normal(self.init_method_std,
                                                    self.num_layers),
      pre_process=self.pre_process,
      post_process=self.post_process,
      args=args
      )
    # 值函数不进行word embedding
    # self.initialize_word_embeddings(init_method_normal,args=args)
      
    if self.post_process:
      self.v_head = ValuleHead(
        args
      )
    #   self.v_head = ParallelLinear(
    #       args.hidden_size,
    #       1,
    #       bias=True,
    #       input_is_parallel=False,
    #       args=args
    # )
    #   self.v_head_layernorm = LayerNorm(self.args.hidden_size, args=self.args)

  
  def reward_loss(self,input_ids,rewards,pad_id):
    """
    计算reward模型的loss
    Args:
        input_ids (_type_): _description_
        rewards (_type_): _description_
        pad_id (_type_): _description_
    """
    chosen_mean_scores = []
    rejected_mean_scores = []

    # Split the inputs and rewards into two parts, chosen and rejected
    assert len(input_ids.shape) == 2
    # print('input_ids.shape',input_ids.shape)
    bs = input_ids.shape[0] // 2
    seq_len = input_ids.shape[1]
    # print('input_ids.shape',input_ids.shape,rewards.shape,bs)
    
    # [batch, maxlen]
    chosen_ids = input_ids[:bs] 
    rejected_ids = input_ids[bs:]
    # [batch, maxlen]
    chosen_rewards = rewards[:bs]
    rejected_rewards = rewards[bs:]

    # print('chosen_rewards',chosen_rewards.shape)
    # print('rejected_rewards',rejected_rewards.shape)
    # print('input_ids',input_ids.shape)

    # Compute pairwise loss. Only backprop on the different tokens before padding
    loss = 0
    for i in range(bs):
      # print(i)
      chosen_id = chosen_ids[i]
      rejected_id = rejected_ids[i]
      chosen_reward = chosen_rewards[i]
      rejected_reward = rejected_rewards[i]
        
      # 最后一个pad
      c_inds = (chosen_id == pad_id).nonzero()
      # 
      c_ind = c_inds[0].item() if len(c_inds) > 0 else seq_len  
      check_divergence = (chosen_id != rejected_id).nonzero()

      if len(check_divergence) == 0:
          end_ind = rejected_reward.size(-1)
          divergence_ind = end_ind - 1
          r_ind = c_ind
      else:
          # Check if there is any padding otherwise take length of sequence
          r_inds = (rejected_id == pad_id).nonzero()
          r_ind = r_inds[0].item(
          ) if len(r_inds) > 0 else seq_len
          end_ind = max(c_ind, r_ind)
          divergence_ind = check_divergence[0]
      assert divergence_ind > 0
      c_truncated_reward = chosen_reward[divergence_ind:end_ind]
      r_truncated_reward = rejected_reward[divergence_ind:end_ind]

      # print('c_ind,r_ind',c_ind,r_ind)
      chosen_mean_scores.append(
          chosen_reward[c_ind - 1])  #use the end score for reference
      rejected_mean_scores.append(rejected_reward[r_ind - 1])
      
      loss += -torch.nn.functional.logsigmoid(c_truncated_reward - r_truncated_reward).mean()
      # loss += -torch.log(
      #     torch.sigmoid(c_truncated_reward - r_truncated_reward)).mean()

    loss = loss / bs
    # chosen_mean_scores = torch.stack(chosen_mean_scores)
    # rejected_mean_scores = torch.stack(rejected_mean_scores)
    return loss
  
  def forward(self,
              input_ids,
              position_ids,
              attention_mask,
              # labels=None,
              # tokentype_ids=None,
              layer_past=None,
              get_key_value=False,
              # forward_method_parallel_output=None,
              # curriculum_seqlen=None,
              reward_model_training=False,
              pad_id=None
              ):

    
    lm_output = self.language_model(input_ids,
                                    position_ids,
                                    attention_mask,
                                    layer_past=layer_past,
                                    get_key_value=get_key_value)
    
    # print('input_ids.shape',input_ids.shape)
    # print('lm_output', lm_output.shape)
    if self.post_process:

      # if self.args.use_v_head_layernorm:
      #   lm_output = self.v_head_layernorm(lm_output)
      v_head_output = self.v_head(lm_output)
      v_head_output = v_head_output.squeeze(-1)
    
      if reward_model_training:
        loss = self.reward_loss(input_ids=input_ids,rewards=v_head_output,pad_id=pad_id)
        return loss
      return v_head_output
    else:
      return lm_output

  
  def forward_eval(self,
              input_ids,
              position_ids,
              attention_mask,
              layer_past=None,
              get_key_value=False,
              pad_id=None,
              prompt_length=0
              ):
    """
    评估阶段的forward函数
    Args:
        input_ids (_type_): _description_
        position_ids (_type_): _description_
        attention_mask (_type_): _description_
        layer_past (_type_, optional): _description_. Defaults to None.
        get_key_value (bool, optional): _description_. Defaults to False.
        pad_id (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    
    lm_output = self.language_model(input_ids,
                                    position_ids,
                                    attention_mask,
                                    layer_past=layer_past,
                                    get_key_value=get_key_value)

    if self.post_process:
      # v_head_output = self.input_layernorm(v_head_output)
      v_head_output = self.v_head(lm_output)
      v_head_output = v_head_output.squeeze(-1)
      # eval_loss
      eval_loss = self.reward_loss(input_ids=input_ids,rewards=v_head_output,pad_id=pad_id)
      
      # chosen and reject score, calculate acc
      # Split the inputs and rewards into two parts, chosen and rejected
      assert len(input_ids.shape) == 2
      # print('input_ids.shape',input_ids.shape)
      bs = input_ids.shape[0] // 2
      seq_len = input_ids.shape[1]
      # print('input_ids.shape',input_ids.shape,rewards.shape,bs)
      # [batch, maxlen]
      chosen_ids = input_ids[:bs] 
      rejected_ids = input_ids[bs:]
      # [batch, maxlen]
      chosen_rewards = v_head_output[:bs]
      rejected_rewards = v_head_output[bs:]

      chosen_end_scores = []
      reject_end_scores = []

      for i in range(bs):
          c_inds = (chosen_ids[i][prompt_length:] == pad_id).nonzero()
          # here we only use the answer part of the sequence so we do not need to care about the padding at the beginning
          c_ind = c_inds[0].item() + prompt_length if len(
              c_inds) > 0 else seq_len

          r_inds = (rejected_ids[i][prompt_length:] == pad_id).nonzero()
          # here we only use the answer part of the sequence so we do not need to care about the padding at the beginning
          r_ind = r_inds[0].item() + prompt_length if len(
              r_inds) > 0 else seq_len
          chosen_end_scores.append(chosen_rewards[i][c_ind - 1].item())
          reject_end_scores.append(rejected_rewards[i][r_ind - 1].item())
      
      return {
        "chosen_end_scores": chosen_end_scores,
        "reject_end_scores": reject_end_scores,
        "loss":eval_loss.item()
      }
    else:
      return lm_output

  
  def forward_value(
    self,
    input_ids,
    position_ids,
    attention_mask,
    # labels=None,
    # tokentype_ids=None,
    layer_past=None,
    get_key_value=False,
    # forward_method_parallel_output=None,
    # curriculum_seqlen=None,
    pad_id=None,
    tokentype_ids=None,
    return_value_only=False,
    prompt_length=0,
    ):
    """
    reward 函数的推理

    Args:
        input_ids (_type_): _description_
        position_ids (_type_): _description_
        attention_mask (_type_): _description_
        layer_past (_type_, optional): _description_. Defaults to None.
        get_key_value (bool, optional): _description_. Defaults to False.
        pad_id (_type_, optional): _description_. Defaults to None.
        return_value_only (bool, optional): _description_. Defaults to False.
        prompt_length (int, optional): _description_. Defaults to 0.

    Returns:
        _type_: _description_
    """

    lm_output = self.language_model(input_ids,
                                    position_ids,
                                    attention_mask,
                                    layer_past=layer_past,
                                    get_key_value=get_key_value)

    if self.post_process:
      # if self.args.use_v_head_layernorm:
      #   lm_output = self.v_head_layernorm(lm_output)
      v_head_output = self.v_head(lm_output)
      v_head_output = v_head_output.squeeze(-1)
      if return_value_only:
        return v_head_output
      
      # print(f'input ids: {input_ids[:40]}')
      # assert prompt_length > 1, "prompt_length must be greater than 1 to help select the end score"
      bs = v_head_output.size(0)
      seq_len = input_ids.shape[1]
      chosen_end_scores = [
      ]  # we use this name for consistency with the original forward function
      for i in range(bs):
          input_id = input_ids[i]
          value = v_head_output[i]
          # print(f'value is:{value[:20]}')
          c_inds = (input_id[prompt_length:] == pad_id).nonzero()
          # here we only use the answer part of the sequence so we do not need to care about the padding at the beginning
          c_ind = c_inds[0].item() + prompt_length if len(
              c_inds) > 0 else seq_len
          # print(f'c index is:{c_ind}')
          chosen_end_scores.append(value[c_ind - 1])
      return {
          "values": v_head_output,
          "chosen_end_scores": torch.stack(chosen_end_scores),
      }

    else:
      return lm_output
    
  def state_dict_for_save_checkpoint(self,
                                     destination=None,
                                     prefix='',
                                     keep_vars=False):

    state_dict_ = {}
    state_dict_[self._language_model_key] \
        = self.language_model.state_dict_for_save_checkpoint(
            destination, prefix, keep_vars)
    # Save word_embeddings.
    if self.post_process and not self.pre_process:
      if hasattr(self,'word_embeddings'):
        state_dict_[self._word_embeddings_for_head_key] \
            = self.word_embeddings.state_dict(destination, prefix, keep_vars)
    if hasattr(self,'v_head'):
      state_dict_['v_head'] = self.v_head.state_dict(destination, prefix, keep_vars)
      # if self.args.use_v_head_layernorm:
      #   state_dict_['v_head_layernorm'] = self.v_head_layernorm.state_dict(destination, prefix, keep_vars)
    return state_dict_

  def load_state_dict(self, state_dict, strict=True):
    """Customized load."""

    # Load word_embeddings.
    if self.post_process and not self.pre_process:
      # print('wd debug',state_dict.keys(),state_dict[self._word_embeddings_for_head_key].keys())
      if hasattr(self,'word_embeddings'):
        self.word_embeddings.load_state_dict(
            state_dict[self._word_embeddings_for_head_key], strict=strict)
    if hasattr(self,'v_head'):
      print('load v_head.........')
      # inference阶段才需要load保存的v-head layernorm
      self.v_head.load_state_dict(state_dict.get('v_head', {}), strict=strict)
      # if self.args.use_v_head_layernorm:
        # self.v_head_layernorm.load_state_dict(state_dict.get('v_head_layernorm', {}), strict=strict)
    if self._language_model_key in state_dict:
      state_dict = state_dict[self._language_model_key]

    self.language_model.load_state_dict(state_dict, strict=strict)
    


# class GPTModelWithValueHead(GPTModel):
#   """临时的值函数

#   Args:
#       GPTModel (_type_): _description_
#   """
#   def __init__(
#       self,
#       num_tokentypes=0,
#       parallel_output=True,
#       pre_process=True,
#       post_process=True,
#       prefix_lm=False,
#       args=None
#   ):
#     # super().__init__(num_tokentypes=num_tokentypes,
#     #                   parallel_output=parallel_output,
#     #                   pre_process=pre_process,
#     #                   post_process=post_process,
#     #                   prefix_lm=False,
#     #                   args=args
#     #                   )

#     MegatronModule.__init__(self)
#     if args is None:
#       args = get_args()
#     # args = get_args()
#     self.parallel_output = parallel_output
#     self.pre_process = pre_process
#     self.post_process = post_process

    
#     self.fp16_lm_cross_entropy = args.fp16_lm_cross_entropy
#     self.init_method_std = args.init_method_std
#     self.num_layers = args.num_layers

#     self.language_model, self._language_model_key = get_language_model(
#         num_tokentypes=num_tokentypes,
#         add_pooler=False,
#         # TODO: Change naming of class from GPT to something that encapsulate prefix lm.
#         encoder_attn_mask_type=AttnMaskType.prefix
#         if prefix_lm else AttnMaskType.causal,
#         init_method=init_method_normal(self.init_method_std),
#         scaled_init_method=scaled_init_method_normal(self.init_method_std,
#                                                      self.num_layers),
#         pre_process=self.pre_process,
#         post_process=self.post_process,
#         args=args
#         )

#     # self.initialize_word_embeddings(init_method_normal,args=args)
          
#     if self.post_process:
#       self.v_head = ParallelLinear(
#           args.hidden_size,
#           1,
#           bias=True,
#           input_is_parallel=False,
#           args=args
#     )
  
#   def forward(self,
#               input_ids,
#               position_ids,
#               attention_mask,
#               labels=None,
#               tokentype_ids=None,
#               layer_past=None,
#               get_key_value=False,
#               forward_method_parallel_output=None,
#               curriculum_seqlen=None):
    
#     lm_output = self.language_model(input_ids,
#                                     position_ids,
#                                     attention_mask,
#                                     layer_past=layer_past,
#                                     get_key_value=get_key_value)

#     if self.post_process:
#       v_head_output = self.v_head(lm_output)
#       # v_head_output = self.input_layernorm(v_head_output)
#       v_head_output = v_head_output.squeeze(-1)
#       return v_head_output
#     else:
#       return lm_output
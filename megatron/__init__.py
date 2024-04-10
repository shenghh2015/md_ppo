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
import torch

from .package_info import (
    __description__,
    __contact_names__,
    __url__,
    __download_url__,
    __keywords__,
    __license__,
    __package_name__,
    __version__,
)
from functools import lru_cache

from .global_vars import get_args
from .global_vars import get_current_global_batch_size
from .global_vars import get_num_microbatches
from .global_vars import update_num_microbatches
from .global_vars import get_tokenizer
from .global_vars import get_tensorboard_writer
from .global_vars import get_adlr_autoresume
from .global_vars import get_timers
from .initialize import initialize_megatron
from megatron import mpu
import torch.distributed as dist
from megatron.monkey_patch import patch_all
import socket
import os
from megatron.filelock import FileLock
import gc 


patch_all()  

def args_get_fn(x,kwargs,final_default=None):
  args = get_args()
  if final_default is None:
    return kwargs.get(x,getattr(args,x))
  return kwargs.get(x,getattr(args,x,final_default))


def print_rank_0(*message):
  """If distributed is initialized, print only on rank 0."""
  if torch.distributed.is_initialized():
    if torch.distributed.get_rank() == 0:
      print(*message, flush=True)
  else:
    print(*message, flush=True)


def is_last_rank():
  return torch.distributed.get_rank() == (torch.distributed.get_world_size() -
                                          1)

def print_with_rank(*message):
  """打印的时候加入rank信息
  """
  rank = dist.get_rank()
  ddp_rank = mpu.get_data_parallel_rank()
  tp_rank = mpu.get_tensor_model_parallel_rank()
  pp_rank = mpu.get_pipeline_model_parallel_rank()
  print(
    f'rank: {rank}, ddp_rank: {ddp_rank}, tp_rank: {tp_rank}, pp_rank: {pp_rank},',
    *message
  )

def print_rank_last(*message):
  """If distributed is initialized, print only on last rank."""
  if torch.distributed.is_initialized():
    if is_last_rank():
      print(*message, flush=True)
  else:
    print(*message, flush=True)


@lru_cache()
def get_local_ip():
  import socket
  return socket.gethostbyname(socket.gethostname())

def tensor_empty_check(input_tensor):
  # result = min(input_tensor.shape, default=0)
  return min(input_tensor.shape) == 0

def __get_custom_loss_mask(tokens):
  tokenizer = get_tokenizer()
  micro_batch_size, seq_length = tokens.size()
  loss_mask = torch.ones(tokens.size(), dtype=torch.float, device=tokens.device)
  loss_mask[tokens == tokenizer.pad] = 0.0
  for b in range(micro_batch_size):
    # Find indecies where EOD token is.
     # 1-d tensor
    eod_index = (tokens[b] == tokenizer.eod).nonzero().squeeze()
    ans_index = (tokens[b] == tokenizer.ans).nonzero().squeeze()
    if not (tensor_empty_check(eod_index) or tensor_empty_check(ans_index)):
        eod_index_list = [{"index":i, "type":"eod"} for i in eod_index.tolist()]
        ans_index_list = [{"index":i, "type":"ans"} for i in ans_index.tolist()]
        eod_ans_index = eod_index_list + ans_index_list
        # min_len = min(len(eod_index, ans_index))
        eod_ans_index.sort(key=lambda x: x["index"])
        eod_ans_list = []
        prev_type = "eod"
        prev_index = 0
        for eod_or_ans in eod_ans_index:
            if eod_or_ans["type"] == "ans" and prev_type == "eod":
                eod_ans_list.append((prev_index, eod_or_ans["index"]))
            prev_type = eod_or_ans["type"]
            prev_index = eod_or_ans["index"]
        if prev_type == "eod":
            eod_ans_list.append((prev_index, loss_mask.shape[1]))
        
        for eod_ans_pair in eod_ans_list:
          loss_mask[b, eod_ans_pair[0]:eod_ans_pair[1]] = 0.0
    
    # remove the ans token
    # loss_mask_wo_ans = loss_mask[b, tokens[b] != tokenizer.ans]
    # padding_seq = torch.tensor(0.0, device=tokens.device).repeat(ans_index.shape[0])
    # loss_mask[b] = torch.cat((loss_mask_wo_ans, padding_seq))
  
  
  return loss_mask

# loss mask must remove ans and prompt
def get_ltor_sft(tokens, tokenizer,reset_attention_mask=True, reset_position_ids=True):
  
  micro_batch_size, seq_length = tokens.size()
  # Loss mask.
  loss_mask = torch.ones(tokens.size(), dtype=torch.float, device=tokens.device)

  # loss_mask_list = []
  # attention_mask_list = []
  
  def pad(cur_tokens, max_len, padding_value):
    assert len(cur_tokens.shape) == 1
    padding_len = max_len - cur_tokens.shape[0]
    if padding_len > 0:
      output_tokens = torch.cat(
        (cur_tokens, 
         torch.tensor(
           [padding_value]*padding_len, device=cur_tokens.device
            ))
        )
      return output_tokens
    return cur_tokens
  
  for b in range(micro_batch_size):
    
    # Find indices where eod and ans is.
    eod_index = (tokens[b] == tokenizer.eod).nonzero().squeeze(1)
    ans_index = (tokens[b] == tokenizer.ans).nonzero().squeeze(1)
    # Prompt loss to 0.
    if not (tensor_empty_check(eod_index) or tensor_empty_check(ans_index)):
      eod_index_list = [{"index":i, "type":"eod"} for i in eod_index.tolist()]
      ans_index_list = [{"index":i, "type":"ans"} for i in ans_index.tolist()]
      eod_ans_index = eod_index_list + ans_index_list
      # min_len = min(len(eod_index, ans_index))
      eod_ans_index.sort(key=lambda x: x["index"])
      eod_ans_list = []
      prev_type = "eod"
      prev_index = 0
      for eod_or_ans in eod_ans_index:
        if eod_or_ans["type"] == "ans" and prev_type == "eod":
            eod_ans_list.append((prev_index, eod_or_ans["index"]))
        prev_type = eod_or_ans["type"]
        prev_index = eod_or_ans["index"]
      if prev_type == "eod":
        eod_ans_list.append((prev_index, loss_mask.shape[1]))
      
      for eod_ans_pair in eod_ans_list:
        # 不屏蔽eod loss，来预测下一个token
        loss_mask[b, eod_ans_pair[0]+1:eod_ans_pair[1]] = 0.0
    tokens_wo_ans_index = (tokens[b] != tokenizer.ans)
    tokens[b] = pad(
      tokens[b][tokens_wo_ans_index], tokens.shape[1], tokenizer.pad)
    loss_mask[b] = pad(
      loss_mask[b][tokens_wo_ans_index], tokens.shape[1], 0.0)
  
  tokens, labels = tokens[:, :-1].contiguous(),tokens[:, 1:].contiguous()
  # loss mask要和label对齐
  loss_mask = loss_mask[:, 1:]
  
  # Reset tokens, micro batch size and sequence length.
  micro_batch_size, seq_length = tokens.size()
  
  # Attention mask.
  attention_mask = torch.tril(
    torch.ones((micro_batch_size, seq_length, seq_length),
                device=tokens.device)).view(micro_batch_size, 1, seq_length,
                                          seq_length)
                
  # Position ids.
  position_ids = torch.arange(seq_length, dtype=torch.long, device=tokens.device)
  position_ids = position_ids.unsqueeze(0).expand_as(tokens)
  
  # Prevent cross document interactions.

  for b in range(micro_batch_size):
    eod_index = (tokens[b] == tokenizer.eod).nonzero().squeeze(1).tolist()
    # # if isinstance(eod_index, int):
    # #   eod_index = [eod_index]
    # if not isinstance(eod_ans_index, list):
    #   eod_index = [eod_index]
    # print(f'eod is :{eod_index}')
    if reset_attention_mask:
      for i in eod_index:
        attention_mask[b, 0, (i + 1):, :(i + 1)] = 0

  # 去掉ans token
    if reset_position_ids:
      prev_index = 0
      for i in eod_index:
        position_ids[b, (i + 1):] -= (i + 1 - prev_index)
        prev_index = i + 1
    
  attention_mask = (attention_mask < 0.5).type(torch.uint8)
  
  return (tokens, position_ids, attention_mask), (labels, loss_mask)


def prompt_template(prompt,ans=""):
  """目前适用于多轮的

  Args:
      prompt (_type_): _description_

  Returns:
      _type_: _description_
  """
  template = "Human:{prompt}\nAssistant:{ans}"
  if isinstance(prompt,str):
    return template.format(prompt=prompt,ans=ans)
  else:
    if ans:
      assert isinstance(prompt,list) and isinstance(ans,list), (type(prompt),type(ans))
      return [template.format(prompt=p,ans=a) for p,a in zip(prompt,ans)]
    else:
      assert ans == "", ans
      return [template.format(prompt=p,ans=ans) for p in prompt]


def remove_prompt_template(prompt):
  def _remove(t):
    return t.replace('Human:',"").replace("\nAssistant:","")
  if isinstance(prompt,str):
    return _remove(prompt)  
  return [_remove(p) for p in prompt]


def get_server_ip():
  return socket.gethostbyname(socket.gethostname())


def torch_save_helper(origin_fn,*args_,**kwargs):
  """
  
  """
  args = get_args()
  # 同一台机器的rank避免同时进行参数保存
  filelock = os.path.join(args.save,f'{get_server_ip()}.save.checkpoint')
  # 默认的timeout是600s
  assert os.path.exists(args.save),args.save
  with FileLock(filelock,30*60):
    # torch.cuda.empty_cache()
    # gc.collect()
    # print_with_rank(f'get lock on :{get_server_ip()}, saving, pid: {os.getpid()}')
    origin_fn(*args_,**kwargs)
    # print_with_rank(f'{get_server_ip()}, saved, pid: {os.getpid()}')
    
      



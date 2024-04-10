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
"""Utilities for generating text."""


import copy
import json
import os
import time
import math

import queue

import torch
import torch.nn.functional as F
from deepspeed.runtime.pipe import schedule
import torch.distributed as dist

from megatron import get_args
from megatron import get_tokenizer
from megatron import mpu
from tqdm import tqdm
from megatron.utils import get_ltor_masks_and_position_ids, unwrap_model
from megatron.p2p_communication import recv_forward, send_forward
from megatron import p2p_communication
from megatron.text_generation_utils import distribute_tokenize

# These are needed to unwrap the model, would be nice to put these in megatron.utils if possible?
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP
from megatron.model import DistributedDataParallel as LocalDDP
from megatron.model import Float16Module
from megatron import get_args
from megatron import print_rank_0,print_with_rank,print_rank_last


def get_batch(context_tokens):
  """Generate batch from context tokens."""
  args = get_args()
  tokenizer = get_tokenizer()

  # Move to GPU.
  # tokens = context_tokens.view(args.micro_batch_size, -1).contiguous().cuda()
  tokens = context_tokens.contiguous().cuda()
  # Get the attention mask and position ids.
  attention_mask, _, position_ids = get_ltor_masks_and_position_ids(
      tokens,
      tokenizer.eod,
      False,
      False,
      args.eod_mask_loss,
      prefix_indices=None,
      loss_on_targets_only=args.loss_on_targets_only)
  # attention_mask = None
  return tokens, attention_mask, position_ids


def top_k_logits(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
  """ This function has been mostly taken from huggingface conversational
     ai code at
         https://medium.com/huggingface/how-to-build-a-state-of-the-art-
              conversational-ai-with-transfer-learning-2d818ac26313 """

  if top_k > 0:
    # Remove all tokens with a probability less than the
    # last token of the top-k
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits[indices_to_remove] = filter_value

  if top_p > 0.0:
    # Cconvert to 1D
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep also the first token
    # above the threshold
    sorted_indices_to_remove[..., 1:] \
        = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    for i in range(sorted_indices.size(0)):
      indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
      logits[i][indices_to_remove] = filter_value

  return logits

# We added this function to support the tasks evaluation such as squad
# and drop in the https://github.com/EleutherAI/lm-evaluation-harness
# codebase. The lm-evaluation-harness code can now call this function
# similar to their current generate function call used for gpt style models.
def generate_samples(
    model,
    contexts:list, 
    max_new_tokens,
    tokenizer,
    eos_token_id,
    max_seq_len,
    min_new_tokens=None,
    recompute=False,
    greedy=False,
    temperature=1,
    top_k=0,
    top_p=1,
  ):
  torch.cuda.empty_cache()
  # Generate samples for lm evaluation
  # NEED TO THINK ABOUT eos token
  assert isinstance(contexts,list), contexts
  # args = get_args()
  # tokenizer = get_tokenizer()
  batch_size = len(contexts)
  raw_text_lens = [len(context) for context in contexts]
  
  model.eval()

  contexts_tokens = [tokenizer.tokenize(context) for context in contexts]

  query_tensors = [torch.LongTensor([context_tokens]) for context_tokens in contexts_tokens]
    
  contexts_tensor_len = [len(context_tokens) for context_tokens in contexts_tokens]
  max_context_tensor_len = max(contexts_tensor_len)
  assert max_context_tensor_len < max_seq_len

  # args.out_seq_length = max_gen_length + len(context_tokens)
  # print(args.out_seq_length,context_tensor_len)
  # args.eos_id = eos_token_id
  # pdb.set_trace()
  # print_with_rank(f'> max_generated_token: {max_new_tokens+max_context_tensor_len}\n' +\
  #                 f'> max_new_token: {max_new_tokens}\n' +\
  #                 f'> max_seq_len: {max_seq_len}\n' +\
  #                 f'> recompute: {recompute}' +\
  #                 f'> greedy: {greedy}' +\
  #                 f'> top_k: {top_k}' +\
  #                 f'> top_p: {top_p}' +\
  #                 f'> tempature: {temperature}'
  #                 )
  with torch.no_grad():
    token_stream = get_token_stream(
      model, 
      contexts_tokens,
      max_generated_len=min(
        max_new_tokens+max_context_tensor_len,max_seq_len
        ),
      min_new_tokens=min_new_tokens,
      max_seq_len=max_seq_len, 
      eos_token_id=eos_token_id,
      recompute=recompute,
      greedy=greedy,
      temperature=temperature,
      top_k=top_k,
      top_p=top_p
      )
    for counter, (decode_tokens,lengths,lm_logits, values) in enumerate(token_stream):
      pass
    # print_with_rank(f'finish generate iteration')
    
          # if counter == args.out_seq_length:
          #     break
  if decode_tokens is None:
    return None

  decode_tokens = decode_tokens.cpu()
  lengths = lengths.cpu()
  values = values.cpu()
  lm_logits = lm_logits.cpu()

  
  # print_with_rank('contexts_tensor_len',contexts_tensor_len)
  # print('values',values.shape)
  # print('lm_logits',lm_logits.shape)
  # print('lengths',lengths)
  new_token_num = []
  for i in range(batch_size):
    # 此时表示只有生成了一个eod,
    if lengths[i] == contexts_tensor_len[i]:
      new_token_num.append(1)
    else:
      new_token_num.append(lengths[i]-contexts_tensor_len[i]+1)  # 避免eod被纳入


  # actual_gened_lens = [min(lengths[i]+1,contexts_tensor_len[i]+max_new_tokens) for i in range(batch_size)]
  response_tensors =[decode_tokens[i,contexts_tensor_len[i]:(contexts_tensor_len[i]+new_token_num[i])].unsqueeze(0) for i in range(batch_size)]
  values = [values[i,contexts_tensor_len[i]-1:(contexts_tensor_len[i]-1+new_token_num[i])].unsqueeze(0) for i in range(batch_size)]
  lm_logits = [lm_logits[i,contexts_tensor_len[i]-1:(contexts_tensor_len[i]-1+new_token_num[i]),:].unsqueeze(0) for i in range(batch_size)]
  # actual_gened_lens = [min(lengths[i]+1,contexts_tensor_len[i]+max_new_tokens) for i in range(batch_size)]
  # response_tensors =[decode_tokens[i,contexts_tensor_len[i]:actual_gened_lens[i]].unsqueeze(0) for i in range(batch_size)]
  # values = [values[i,contexts_tensor_len[i]-1:actual_gened_lens[i]-1].unsqueeze(0) for i in range(batch_size)]
  # lm_logits = [lm_logits[i,contexts_tensor_len[i]-1:actual_gened_lens[i]-1,:].unsqueeze(0) for i in range(batch_size)]
  

  # 
  # print_with_rank('contexts_tensor_len',contexts_tensor_len)
  # print('values',values.shape)
  # print('lm_logits',lm_logits.shape)
  # print('lengths',lengths)
  
  # decode_tokens[:,context_tensor_len:]
  # lm_logits = lm_logits[:,context_tensor_len-1:-1,:]
  # values = values[:,context_tensor_len:,:]
  # decode_tokens, _ = decode_tokens
  # decode_tokens = decode_tokens.cpu().numpy().tolist()
  trim_decode_tokens = []
  for i in range(batch_size):
    trim_decode_tokens.append(tokenizer.detokenize(
        decode_tokens[i,:(contexts_tensor_len[i]+new_token_num[i])].tolist())[raw_text_lens[i]:]
    )
    # logits_max = lm_logits[i].topk(3,dim=-1)
    # logits_max = 
    # print('logits max',logits_max.values,logits_max.indices)

  # print('query_tensors lens',len(query_tensors),len(response_tensors))
  return {
      "prompts":contexts,
      "query_tensors": query_tensors,
      "response_tensors": response_tensors,
      'values': values,
      'lm_logits': lm_logits,
      'generated_texts': trim_decode_tokens   
  }


def generate_samples_with_pipeline(
    model, 
    prompts, 
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
    top_p=1
):
  """支持以pipeline的方式进行采样,适用于跑批量任务，
  主要实现的步骤为:

  Args:
      model (_type_): _description_
      prompts (_type_): _description_
      max_new_tokens (_type_): _description_
      tokenizer (_type_): _description_
      eos_token_id (_type_): _description_
      max_seq_len (_type_): _description_
      recompute (bool, optional): _description_. Defaults to False.
      greedy (bool, optional): _description_. Defaults to False.
      temperature (int, optional): _description_. Defaults to 1.
      top_k (int, optional): _description_. Defaults to 0.
      top_p (int, optional): _description_. Defaults to 1.
      pipeline_batch_size (_type_, optional): _description_. pipeline_batch_size表示流水线的batch size
      micro_batch_size 表示多少个样本一组
  """
  
  def _pad(batch,pad_id,maxlen=None):
    if maxlen is None:
      lens = [len(tokens) for tokens in batch]
      maxlen = max(lens)
    context_lengths = []
    for tokens in batch:
      context_length = len(tokens)
      if context_length < maxlen:
        tokens.extend([pad_id] * (maxlen - context_length))
      context_lengths.append(context_length)
    return batch, context_lengths


  def _generate_position_and_attention_mask(input_ids:torch.Tensor):
    # attention_mask与position
    batch_size,maxlen = input_ids.shape
    attention_mask = torch.tril(
    torch.ones((batch_size, maxlen, maxlen),
                device=input_ids.device)).view(batch_size, 1, maxlen,
                                          maxlen)
    # Position ids.
    position_ids = torch.arange(
      maxlen, dtype=torch.long, device=input_ids.device)
    position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
    attention_mask = (attention_mask < 0.5)
    return attention_mask,position_ids
  
  def _get_queue_num_element(q:queue.Queue,num):
    ret = []
    for _ in range(num):
      if q.empty():
        break
      ret.append(q.get()[1])
    return ret
  
  def _sampling(task):
    # 采样函数
    if mpu.is_pipeline_last_stage():
      # last pipeline stage
      context_length = task['context_length']
      logits = task['logits'][:, context_length - 1, :].clone()
      tokens = task['input_ids']
      lengths = task['lengths']
      if min_new_tokens is not None:
        # 将eod位置的logits重置为-inf,避免过早的采样到eod
        is_reach_min_new_token = (context_length - context_lengths) < min_new_tokens
        logits[is_reach_min_new_token,eos_token_id] = -math.inf

      if greedy:
        prev = torch.argmax(logits, dim=-1).view(-1)
      else:
        logits = logits.float()
        #TODO 加入random温度函数
        logits /= temperature
        logits = top_k_logits(logits, top_k=top_k, top_p=top_p)
        log_probs = F.softmax(logits, dim=-1)
        prev = torch.multinomial(log_probs, num_samples=1).view(-1)  # sampling
      started = (task['context_lengths'] <= task['context_length']).cuda()
      # if context_length 
      # tokens: b x max_len
      # contenxt = mex_len (invlid)
      new_tokens = switch(tokens[:, context_length].view(-1).cuda(), prev, started)
      # 将新的token写入到input_ids中
      if context_length >= tokens.shape[1]:
        # 此时需要进行将新的token拼接到后面
        tokens = torch.cat((tokens,new_tokens.unsqueeze(-1)),dim=-1)
      else:
        tokens[:, context_length] = new_tokens
      task['input_ids'] = tokens
      # 将new token发送到first stage中
      src = mpu.get_pipeline_model_parallel_last_rank()
      group = mpu.get_embedding_group()
      torch.distributed.broadcast(new_tokens, src, group)
      # 判断当前任务是否结束,并将结束符发送到其他rank
      is_done = task['done_tensor']
      done_token = (prev == eos_token_id).byte() & started.byte()
      just_finished = (done_token & ~is_done).bool()
      lengths[just_finished.view(-1)] = context_length
      is_done = is_done | done_token

      task['done_tensor'] = is_done
    
      done = torch.all(is_done)   # whether all generation tasks are done
      src = mpu.get_pipeline_model_parallel_last_rank()
      group = mpu.get_pipeline_model_parallel_group()
      torch.distributed.broadcast(done, src, group)
      task['is_done'] = done
    else:
      # first pipeline stage
      if mpu.is_pipeline_first_stage():
        context_length = task['context_length']
        tokens = task['input_ids']
        src = mpu.get_pipeline_model_parallel_last_rank()
        group = mpu.get_embedding_group()
        new_tokens = torch.empty_like(tokens[:, context_length]).cuda()
        torch.distributed.broadcast(new_tokens, src, group)
        # 将新的token写入到input_ids中
        if context_length > tokens.shape[1]:
          # 此时需要进行将新的token拼接到后面
          tokens = torch.cat((tokens,new_tokens.unsqueeze(-1)),dim=-1)
        else:
          tokens[:, context_length] = new_tokens
        task['input_ids'] = tokens

      done = torch.cuda.ByteTensor([0])
      src = mpu.get_pipeline_model_parallel_last_rank()
      group = mpu.get_pipeline_model_parallel_group()
      torch.distributed.broadcast(done, src, group)
      task['is_done'] = done

    # token generation is done and context_length is updated
    task['context_length'] += 1
    
    # 此时超过了序列的最大长度，需要退出
    if task['context_length']  >= task["max_seq_len"]:
      task['is_done'] = True

  args = get_args()
  args.variable_seq_lengths = True
  tokenizer = get_tokenizer()
  if isinstance(prompts,str):
    prompts = [prompts]
  
  assert len(prompts) >0, prompts
  raw_text_lens = [len(prompt) for prompt in prompts]

  # 分布式tokenizer
  tokenized_prompts = distribute_tokenize(prompts,tokenizer,args.seq_length)
  assert len(tokenized_prompts) == len(prompts), f'tokenized_prompts: {len(tokenized_prompts)}, prompts: {len(prompts)}'

  # pipeline_batches = []
  # 任务构造，数据暂时不加载到gpu, 减小显存
  tasks_queue = queue.PriorityQueue()
  for i,batch_start in enumerate(range(0,len(prompts),micro_batch_size)):
    prompts_batch = prompts[batch_start:batch_start+micro_batch_size]
    tokenized_prompts_batch = tokenized_prompts[batch_start:batch_start+micro_batch_size]
    raw_text_lens_batch = raw_text_lens[batch_start:batch_start+micro_batch_size]
    input_ids, context_lengths = _pad(
      tokenized_prompts_batch,tokenizer.pad,
      maxlen=args.seq_length
      )
    input_ids = torch.LongTensor(input_ids)
    context_lengths =  torch.LongTensor(context_lengths)
    attention_mask,position_ids = _generate_position_and_attention_mask(
      input_ids
      )
    context_length = context_lengths.min().item()
    max_context_length = context_lengths.max().item()
    if min_new_tokens is not None:
      min_new_tokens_ = min_new_tokens[batch_start:batch_start+micro_batch_size].cuda()
    else:
      min_new_tokens_ = None
    max_seq_len_ = min(
       max_context_length + max_new_tokens, max_seq_len)
    
    # print_with_rank('**********'\
    #                 f'>> micro_batch_size: {micro_batch_size},prompt_batch: {len(prompts_batch)}, input_ids: {len(input_ids), input_ids[0].size()}, context_lengths: {context_lengths}, '\
    #                 f'attention: {attention_mask.size()}, position_ids: {position_ids.size()}'\
    #                 f'min context_length: {context_length}, max_new_tokens: {max_new_tokens}, min_new_tokens: {min_new_tokens_} '\
    #                 f'done_tensor: {torch.zeros([input_ids.shape[0]]).byte().cuda().size()}'\
    #                 f'pipeline_batch_size:{pipeline_batch_size}, max_seg_len_: {max_seq_len_}')

    task = {
        "prompts_batch":prompts_batch,
        "input_ids":input_ids,
        "position_ids": position_ids,
        "attention_mask": attention_mask,
        "task_id": i,
        "tokentype_ids":None,
        "layer_past": None,
        "origin_context_length":context_length,
        "context_length": context_length,
        "context_lengths": context_lengths,
        "new_token_num":0,
        "raw_text_lens":raw_text_lens_batch,
        "max_seq_len": max_seq_len_,
        "min_new_tokens":min_new_tokens_,
        "values": None,
        "done_tensor": torch.zeros([input_ids.shape[0]]).byte().cuda(), 
        "is_done": False, 
        "logits": None,
        "lengths":torch.ones([input_ids.shape[0]]).long() * max_seq_len_
      }
    tasks_queue.put((i,task))
  
  # 实例化任务队列
  ret_queue = queue.PriorityQueue()
  model.eval()
  if pipeline_batch_size is None:
    pipeline_batch_size = tasks_queue.qsize()
  if dist.get_rank() == 0:
    pbar = tqdm(total=tasks_queue.qsize())

  with torch.no_grad():
    while not tasks_queue.empty():
      # 从队列中取出任务
      tasks = _get_queue_num_element(
        tasks_queue,pipeline_batch_size
        )
      # 模型计算
      ppo_forward_step_with_pipelining(
        model,
        tasks,
        forward_method_parallel_output=False
      )
      # 生成采样, 没有结束的任务继续送到任务队列中
      # TODO
      for task in tasks:
        _sampling(task)
        if task['is_done']:
          # 当前任务结束,进行后处理
          ret_queue.put(
            (
              task['task_id'],
              {
              "task_id": task['task_id'],
              "input_ids": task['input_ids'].cpu(),
              "values": task['values'].cpu() if task['values'] is not None else None,
              "logits": task['logits'].cpu() if task['logits'] is not None else None,
              "lengths": task['lengths'].cpu() if task['lengths'] is not None else None,
              "raw_text_lens": task['raw_text_lens'],
              "context_lengths": task['context_lengths']
              }
            )
          )
          del task
          if dist.get_rank() == 0:
            pbar.update(1)
          
          torch.cuda.empty_cache()
        else:
          tasks_queue.put((task['task_id'],task))

    # 格式化返回的结果
    ret = {
      "query_tensors": [],
      "response_tensors": [],
      'values': [],
      'lm_logits': [],
      'generated_texts': []
    }
    # 只有最后一个rank才进行计算
    if not (mpu.is_pipeline_last_stage() \
        and mpu.get_tensor_model_parallel_rank() == 0):
      return None
    while not ret_queue.empty():
      task = ret_queue.get()[1]
      decode_tokens = task['input_ids']
      lengths = task['lengths']
      raw_text_lens = task['raw_text_lens']
      values = task['values']
      lm_logits = task['logits']
      contexts_tensor_len = task['context_lengths']
      batch_size = decode_tokens.shape[0]

      new_token_num = []
      for i in range(batch_size):
        # 此时表示只有生成了一个eod
        if lengths[i] == contexts_tensor_len[i]:
          new_token_num.append(1)
        else:
          new_token_num.append(lengths[i]-contexts_tensor_len[i]+1)  # 避免eod被纳入
      # 
      response_tensors =[decode_tokens[i,contexts_tensor_len[i]:(contexts_tensor_len[i]+new_token_num[i])].unsqueeze(0) for i in range(batch_size)]
      query_tensors = [decode_tokens[i,:contexts_tensor_len[i]].unsqueeze(0) for i in range(batch_size)]
      values = [values[i,contexts_tensor_len[i]-1:(contexts_tensor_len[i]-1+new_token_num[i])].unsqueeze(0) for i in range(batch_size)]
      lm_logits = [lm_logits[i,contexts_tensor_len[i]-1:(contexts_tensor_len[i]-1+new_token_num[i]),:].unsqueeze(0) for i in range(batch_size)]
       
      trim_decode_tokens = []
      for i in range(batch_size):
        trim_decode_tokens.append(tokenizer.detokenize(
            decode_tokens[i,:(contexts_tensor_len[i]+new_token_num[i])].tolist())[raw_text_lens[i]:]
        )
      
      ret["query_tensors"].extend(query_tensors)
      ret["response_tensors"].extend(response_tensors)
      ret['values'].extend(values)      
      ret['lm_logits'].extend(lm_logits)
      ret['generated_texts'].extend(trim_decode_tokens)
  
  return ret

# dump_tensor = None


class LoggingRank(object):

  def __init__(self, msg):
    self.rank = dist.get_rank()
    self.msg = msg
    self.pid = os.getpid()

  def __enter__(self):
    print(f'rank: {self.rank}, pid: {self.pid}, enter {self.msg}')
    return

  def __exit__(self, exc_type, exc_val, exc_tb):
    print(f'rank: {self.rank}, pid: {self.pid}, exit {self.msg}')


def is_pipeline_first_or_last_stage():
  return mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage()


def pad_batch(batch, pad_id, max_seq_len):

  context_lengths = []
  for tokens in batch:
    context_length = len(tokens)
    if context_length < max_seq_len:
      tokens.extend([pad_id] * (max_seq_len - context_length))
    context_lengths.append(context_length)
  return batch, context_lengths


def get_token_stream(
  model, 
  context_tokens,
  max_generated_len,
  eos_token_id,
  max_seq_len,
  min_new_tokens=None,
  recompute=False,
  greedy=False,
  temperature=1,
  top_k=0,
  top_p=1,
  ):
  """_summary_

  Args:
      model (_type_): _description_
      context_tokens (_type_): _description_
      max_generated_len (_type_): 生成后的最大长度
      eos_token_id (_type_): _description_
      max_seq_len (_type_): 模型设定的序列最大token数目

  Yields:
      _type_: _description_
  """
  from megatron import print_with_rank
  # args = get_args()
  # tokenizer = get_tokenizer()
  # tensor强制padding到max_seq_len，这个是由于megatron模型导致的。
  # TODO: 修改此处的强制padding
  context_tokens, context_lengths = pad_batch(context_tokens, eos_token_id,
                                              max_seq_len)

  # print(f'padding 之后的context: {context_tokens}')

  context_tokens_tensor = torch.cuda.LongTensor(context_tokens)
  context_length_tensor = torch.cuda.LongTensor(context_lengths)
  
  # print_with_rank('context_length_tensor',context_length_tensor)
  torch.distributed.broadcast(context_length_tensor,
                              mpu.get_tensor_model_parallel_src_rank(),
                              group=mpu.get_tensor_model_parallel_group())
  torch.distributed.broadcast(context_tokens_tensor,
                              mpu.get_tensor_model_parallel_src_rank(),
                              group=mpu.get_tensor_model_parallel_group())

  # print_with_rank('after context_length_tensor',context_length_tensor)
  
  context_length = context_length_tensor.min().item()
  tokens, attention_mask, position_ids = get_batch(context_tokens_tensor)
  # print_with_rank('start sample_sequence_batch')
  batch_token_iterator = sample_sequence_batch(model,
                                               context_tokens_tensor,
                                               context_length_tensor,
                                               attention_mask,
                                               position_ids,
                                               eos_token_id,
                                               max_generated_len,
                                               min_new_tokens=min_new_tokens,
                                               recompute=recompute,
                                               greedy=greedy,
                                               temperature=temperature,
                                               top_k=top_k,
                                               top_p=top_p,
                                               )
  for i,(tokens, lengths, lm_logits, values) in enumerate(batch_token_iterator):
    # print_with_rank(f'finish sample_sequence_batch {i}')
    context_length += 1
    if tokens is not None and lm_logits is not None:
      # if lm_logits is not None:
      #   print('lengths shape', lm_logits.shape)
      yield tokens, lengths, lm_logits, values
    else:
      yield None, None, None, None


def switch(val1, val2, boolean):

  boolean = boolean.type_as(val1)
  return (1 - boolean) * val1 + boolean * val2


def forward_step(model,
                 tokens,
                 position_ids,
                 attention_mask,
                 tokentype_ids,
                 layer_past=None,
                 get_key_value=None,
                 forward_method_parallel_output=None):

  # Hidden size changes when not using recompute, need to tell p2p_communicate
  # functions the correct size
  args = get_args()
  orig_seq_length = args.seq_length
  args.seq_length = tokens.shape[1]

  input_tensor = recv_forward()

  # Forward pass through the model.
  unwrapped_model = unwrap_model(model, (torchDDP, LocalDDP, Float16Module))
  
  # print('sfgsfhg',type(unwrapped_model),type(unwrapped_model.module),dir(unwrapped_model.module))
  # print('sdsfb',unwrapped_model.module.modules())
  # for m in unwrapped_model.module.modules():
  #     print(type(m) )
  # attention_mask = None
  unwrapped_model.set_input_tensor(input_tensor)
  output_tensor = model(
      tokens,
      position_ids,
      attention_mask,
      tokentype_ids=tokentype_ids,
      layer_past=layer_past,
      get_key_value=get_key_value,
      forward_method_parallel_output=forward_method_parallel_output)

  if get_key_value:
    output_tensor, layer_past = output_tensor

  send_forward(output_tensor)

  args.seq_length = orig_seq_length
  if get_key_value:
    return output_tensor, layer_past
  return output_tensor




def ppo_forward_step_with_pipelining(
  model,
  tasks,
  forward_method_parallel_output=False
  ):
  """
  pipeline forward 算法的实现
  """
  unwrapped_model = unwrap_model(model, (torchDDP, LocalDDP, Float16Module))

  # Run warmup forward passes.
  for task in tasks:
    input_tensor = p2p_communication.recv_forward()
    unwrapped_model.set_input_tensor(input_tensor)
    context_length = task['context_length']
    input_ids = task['input_ids']
    position_ids = task['position_ids']
    attention_mask_use = task['attention_mask']
    layer_past = task['layer_past']
    if layer_past is None:
      tokens2use = input_ids[:, :context_length]
      positions2use = position_ids[:, :context_length]
      layer_past_use = None
    else:
      tokens2use = input_ids[:, context_length - 1].unsqueeze(-1)
      positions2use = position_ids[:, context_length - 1].unsqueeze(-1)
      layer_past_use = layer_past
  
    output_tensor = unwrapped_model(
      tokens2use.cuda().contiguous(), 
      positions2use.cuda().contiguous(),
      attention_mask_use.cuda().contiguous(),
      tokentype_ids=None,
      layer_past=layer_past_use,
      get_key_value=True,
      forward_method_parallel_output=forward_method_parallel_output
      )
    
    output_tensor, layer_past = output_tensor
    task['layer_past'] = layer_past
    
    p2p_communication.send_forward(output_tensor)

    if mpu.is_pipeline_last_stage():
      # sampling
      origin_logits_, values_ = output_tensor
      if task['logits'] is not None:
        task['logits'] = torch.cat((task['logits'],origin_logits_),dim=1)
      else:
        task['logits'] = origin_logits_
      
      if task['values'] is not None:
        task['values'] = torch.cat((task['values'],values_),dim=1)
      else:
        task['values'] = values_

def sample_sequence_batch(model,
                          context_tokens,
                          context_lengths,
                          attention_mask,
                          position_ids,
                          eos_id,
                          maxlen,
                          min_new_tokens=None,
                          greedy=False,
                          temperature=1,
                          top_k=0,
                          top_p=1,
                          type_ids=None,
                          recompute=False
                          ):

  # args = get_args()
  from megatron import print_with_rank
  # tokenizer = get_tokenizer()
  model.eval()
  with torch.no_grad():
    context_length = context_lengths.min().item()
    # added eos_id to support the function generate_samples_eval that passes
    # eos_id as an argument and needs termination when that id id found.
    # if hasattr(args, 'eos_id'):
    #   eos_id = args.eos_id
    # else:
    #   eos_id = tokenizer.eod
    counter = 0
    # org_context_length = context_length
    layer_past = None
    batch_size = context_tokens.size(0)
    is_done = torch.zeros([batch_size]).byte().cuda()
    tokens = context_tokens
    origin_logits = None
    values = None
    # if maxlen is None:
    #   maxlen = args.seq_length - 1
    #   if maxlen > (org_context_length + args.out_seq_length):
    #     maxlen = org_context_length + args.out_seq_length
  
    lengths = torch.ones([batch_size]).long().cuda() * maxlen

    while context_length < maxlen:
      # print_with_rank('context_len',context_length)
      # dist.barrier()
      # print_rank_0(f'run : context_length: {context_length}')
      if recompute:
        # print('tokens.shape',tokens.shape)
        output = forward_step(model,
                              tokens[:, :context_length],
                              position_ids[:, :context_length],
                              attention_mask,
                              tokentype_ids=type_ids,
                              forward_method_parallel_output=False)
        # print_with_rank(f'tokens.shape after forward: {context_length}',tokens.shape)
        
        # if mpu.is_pipeline_last_stage():
        #     assert output is not None
        #     origin_logits,values = output
        #     logits,values = origin_logits[:, context_length - 1, :]
      else:
        types2use = None
        if counter == 0:
          tokens2use = tokens[:, :context_length]
          positions2use = position_ids[:, :context_length]
          if type_ids is not None:
            types2use = type_ids[:, :context_length]
        else:
          tokens2use = tokens[:, context_length - 1].view(batch_size, -1)
          positions2use = position_ids[:, context_length - 1].view(
              batch_size, -1)
          if type_ids is not None:
            types2use = type_ids[:, context_length - 1].view(batch_size, -1)
        output, layer_past = forward_step(model,
                                          tokens2use,
                                          positions2use,
                                          attention_mask,
                                          layer_past=layer_past,
                                          get_key_value=True,
                                          tokentype_ids=types2use,
                                          forward_method_parallel_output=False)
        # if mpu.is_pipeline_last_stage():
        #     assert output is not None
        #     logits = output[:, -1].view(batch_size, -1).contiguous()

      if mpu.is_pipeline_last_stage():
        assert output is not None
        origin_logits_, values_ = output
        # 不进行recompute的时候，origin_logits仅仅输出1个位置的logits
        if origin_logits is not None and not recompute:
          origin_logits = torch.cat((origin_logits,origin_logits_),dim=1)
        else:
          origin_logits = origin_logits_
        
        if values is not None and not recompute:
          values = torch.cat((values,values_),dim=1)
        else:
          values = values_

        # print(origin_logits)
        # if not recompute:
        #   origin_logits, presents = origin_logits
        # print('origin_logits shape', origin_logits.shape)
        # print(origin_logits.shape)
        logits = origin_logits[:, context_length - 1, :].clone()
        
        if min_new_tokens is not None:
          # 将eod位置的logits重置为-inf,避免过早的采样到eod
          is_reach_min_new_token = (context_length - context_lengths) < min_new_tokens
          logits[is_reach_min_new_token,eos_id] = -math.inf

        if greedy:
          prev = torch.argmax(logits, dim=-1).view(-1)
        else:
          logits = logits.float()
          logits /= temperature
          logits = top_k_logits(logits, top_k=top_k, top_p=top_p)

          log_probs = F.softmax(logits, dim=-1)
          prev = torch.multinomial(log_probs, num_samples=1).view(-1)
        
        # 判断当前的序列是否开始正式生成
        started = context_lengths <= context_length
        # 如果当前序列没有开始就用原始的token替换
        new_tokens = switch(tokens[:, context_length].view(-1), prev, started)
        tokens[:, context_length] = new_tokens
        src = mpu.get_pipeline_model_parallel_last_rank()
        group = mpu.get_embedding_group()
        torch.distributed.broadcast(new_tokens, src, group)
        # 判断当前序列是否生成结束
        done_token = (prev == eos_id).byte() & started.byte()
        just_finished = (done_token & ~is_done).bool()
        # lengths 变量记录了当前序列退出时的长度
        lengths[just_finished.view(-1)] = context_length
        is_done = is_done | done_token

        done = torch.all(is_done)
        src = mpu.get_pipeline_model_parallel_last_rank()
        group = mpu.get_pipeline_model_parallel_group()
        torch.distributed.broadcast(done, src, group)
        yield tokens[:, :context_length+1], lengths, \
          origin_logits[:, :context_length, :], values[:, :context_length] 

      else:
        if mpu.is_pipeline_first_stage():
          src = mpu.get_pipeline_model_parallel_last_rank()
          group = mpu.get_embedding_group()
          new_tokens = torch.empty_like(tokens[:, context_length])
          torch.distributed.broadcast(new_tokens, src, group)
          tokens[:, context_length] = new_tokens
          yield tokens[:, :context_length+1], None,None,None
        else:
          yield None, None, None, None
        
        done = torch.cuda.ByteTensor([0])
        src = mpu.get_pipeline_model_parallel_last_rank()
        group = mpu.get_pipeline_model_parallel_group()
        torch.distributed.broadcast(done, src, group)

      context_length += 1
      counter += 1
      if done:
        break

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
from megatron import prompt_template

import torch
import torch.nn.functional as F
from deepspeed.runtime.pipe import schedule
import torch.distributed as dist
from tqdm import tqdm

from tqdm import tqdm
from megatron import get_args
from megatron import get_tokenizer
from megatron import mpu
from megatron.utils import get_ltor_masks_and_position_ids, unwrap_model
from megatron.p2p_communication import recv_forward, send_forward

# These are needed to unwrap the model, would be nice to put these in megatron.utils if possible?
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP
from megatron.model import DistributedDataParallel as LocalDDP
from megatron.model import Float16Module
from megatron import get_args
from megatron import print_rank_0

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


def get_batch_pipe(data):
  """Modification of `get_batch` to work on `next(data_iterator)` instead of `data_iterator`"""
  args = get_args()
  tokenizer = get_tokenizer()

  # Items and their type.
  keys = ['text']
  datatype = torch.int64

  # Broadcast data.
  data_b = mpu.broadcast_data(keys, data, datatype)

  # Unpack.
  tokens_ = data_b['text'].long()
  labels = tokens_[:, 1:].contiguous()
  tokens = tokens_[:, :-1].contiguous()

  # Get the masks and position ids.
  attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
      tokens,
      tokenizer.eod,
      args.reset_position_ids,
      args.reset_attention_mask,
      args.eod_mask_loss,
      prefix_indices=None,
      loss_on_targets_only=args.loss_on_targets_only)
  if args.curriculum_learning and args.curriculum_seqlen < tokens.size()[1]:
    # seqlen-based curriculum learning
    # tokens, position_ids, labels, loss_mask have size [batch size, seqlen]
    tokens = tokens[:, :args.curriculum_seqlen].contiguous()
    position_ids = position_ids[:, :args.curriculum_seqlen].contiguous()
    labels = labels[:, :args.curriculum_seqlen].contiguous()
    loss_mask = loss_mask[:, :args.curriculum_seqlen].contiguous()

  print_rank_0(f'loss_mask_sum: {loss_mask.sum(dim=-1)}')
  return (tokens, position_ids, attention_mask), (labels, loss_mask)


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


def generate_samples_input_from_file(model):

  args = get_args()
  tokenizer = get_tokenizer()

  # Read the sample file and open the output file.
  assert args.sample_input_file is not None, \
      'sample input file is not provided.'
  if mpu.is_pipeline_first_stage() and mpu.get_tensor_model_parallel_rank(
  ) == 0:
    fname = open(args.sample_input_file, "r")
    all_raw_text = fname.readlines()
    input_count = len(all_raw_text)
    input_pos = 0
    if args.sample_output_file is None:
      sample_output_file = args.sample_input_file + ".out"
      print('`sample-output-file` not specified, setting '
            'it to {}'.format(sample_output_file))
    else:
      sample_output_file = args.sample_output_file
    fname_out = open(sample_output_file, "w+")

  context_count = 0
  model.eval()
  with torch.no_grad():
    while True:
      terminate_runs = 0
      raw_text_len = 0

      if mpu.is_pipeline_first_stage() \
         and mpu.get_tensor_model_parallel_rank() == 0:
        raw_text = all_raw_text[input_pos]
        input_pos += 1
        if input_pos == input_count:
          raw_text = "stop"
        raw_text_len = len(raw_text)

        if "stop" in raw_text:
          terminate_runs = 1
        else:
          context_tokens = tokenizer.tokenize(raw_text)
          context_length = len(context_tokens)

          # print(f'正在预测:{raw_text} {context_tokens}')

          if context_length >= (args.seq_length // 2):
            print("\nContext length",
                  context_length, "\nPlease give smaller context (half of the "
                  "sequence length)!",
                  flush=True)
            continue
      else:
        context_tokens = tokenizer.tokenize("EMPTY TEXT")
        context_length = 0

      input_info = [terminate_runs, raw_text_len, context_length]
      input_info_tensor = torch.cuda.LongTensor(input_info)

      # 同步信息
      torch.distributed.all_reduce(input_info_tensor,
                                   group=mpu.get_model_parallel_group())
      terminate_runs = input_info_tensor[0].item()
      raw_text_len = input_info_tensor[1].item()
      context_length = input_info_tensor[2].item()

      if terminate_runs == 1:
        return

      # For pipeline parallel we send context tokens to other stages
      # so they get the lengths correct
      if mpu.get_tensor_model_parallel_rank() == 0 \
         and args.pipeline_model_parallel_size > 1:
        if mpu.is_pipeline_first_stage():
          src = mpu.get_pipeline_model_parallel_first_rank()
          group = mpu.get_pipeline_model_parallel_group()
          context_tokens_tensor = torch.cuda.LongTensor(context_tokens)
          torch.distributed.broadcast(context_tokens_tensor, src, group)
        else:
          src = mpu.get_pipeline_model_parallel_first_rank()
          group = mpu.get_pipeline_model_parallel_group()
          context_tokens_tensor = torch.empty(context_length,
                                              dtype=torch.int64,
                                              device=torch.device("cuda"))
          torch.distributed.broadcast(context_tokens_tensor, src, group)
          context_tokens = context_tokens_tensor.cpu().numpy().tolist()

      token_stream = get_token_stream(model, [context_tokens])
      for _, decode_tokens in enumerate(token_stream):
        pass

      if mpu.get_tensor_model_parallel_rank() == 0:
        if mpu.is_pipeline_first_stage():
          # os.system('clear')
          print("\nContext:", raw_text, flush=True)

          fname_out.write("\nContext:")
          fname_out.write(raw_text)

          decode_tokens, _ = decode_tokens
          decode_tokens = decode_tokens[0].cpu().numpy().tolist()
          trim_decode_tokens = tokenizer.detokenize(
              decode_tokens)[raw_text_len:]
          print("\nMegatron-LM:", trim_decode_tokens, flush=True)

          fname_out.write("\n\nMegatron-LM:")
          fname_out.write(trim_decode_tokens)
          fname_out.write("\n")

      raw_text = None
      context_count += 1


# def generate_samples_input_from_file_ds(model):
#     """
#     使用deepspeed进行评估
#     """
#     args = get_args()
#     tokenizer = get_tokenizer()

#     # Read the sample file and open the output file.
#     assert args.sample_input_file is not None, \
#         'sample input file is not provided.'
#     if mpu.is_pipeline_first_stage() and mpu.get_tensor_model_parallel_rank() == 0:
#         fname = open(args.sample_input_file, "r")
#         all_raw_text = fname.readlines()
#         input_count = len(all_raw_text)
#         input_pos = 0
#         if args.sample_output_file is None:
#             sample_output_file = args.sample_input_file + ".out"
#             print('`sample-output-file` not specified, setting '
#                   'it to {}'.format(sample_output_file))
#         else:
#             sample_output_file = args.sample_output_file
#         fname_out = open(sample_output_file, "w+")

#     context_count = 0
#     for prompt in all_raw_text:


# We added this function to support the tasks evaluation such as squad
# and drop in the https://github.com/EleutherAI/lm-evaluation-harness
# codebase. The lm-evaluation-harness code can now call this function
# similar to their current generate function call used for gpt style models.
def generate_samples_eval(model, context, max_gen_length, eos_token_id):
  # Generate samples for lm evaluation
  # NEED TO THINK ABOUT eos token

  args = get_args()
  tokenizer = get_tokenizer()

  raw_text_len = len(context)
  model.eval()

  context_tokens = tokenizer.tokenize(context)
  args.out_seq_length = max_gen_length + len(context_tokens)
  args.eos_id = eos_token_id

  with torch.no_grad():
    token_stream = get_token_stream(model, [context_tokens])
    for counter, decode_tokens in enumerate(token_stream):
      if counter == args.out_seq_length:
        break

  decode_tokens, _ = decode_tokens
  decode_tokens = decode_tokens[0].cpu().numpy().tolist()
  trim_decode_tokens = tokenizer.detokenize(decode_tokens)[raw_text_len:]

  return trim_decode_tokens


# dump_tensor = None


class LoggingRank(object):

  def __init__(self, msg):
    self.rank = dist.get_rank()
    self.msg = msg
    self.pid = os.getpid()

  def __enter__(self):
    print(f'rank: {self.rank}, pid: {self.pid}, enter {self.msg}\n')
    return

  def __exit__(self, exc_type, exc_val, exc_tb):
    print(f'rank: {self.rank}, pid: {self.pid}, exit {self.msg}\n')


def generate_one_token_ds(model, data_iter, batch_id):
  args = get_args()
  model.module.eval()
  # 设置当前的数据集
  model._compute_loss = False  # 推理阶段不计算loss
  model.eval_return_logits = True  # 返回 logits
  model.set_dataiterator(data_iter)
  sched = schedule.InferenceSchedule(micro_batches=1,
                                     stages=model.num_stages,
                                     stage_id=model.stage_id)

  # prevent dead-lock with multiple evals sequence
  # with LoggingRank(f'batch_id: {batch_id}, head barrier'):
  dist.barrier()
  # print_rank_0(f'fist stage start _exec_schedule, batch_id: {batch_id}')
  # 执行计算
  with torch.no_grad():
    model._exec_schedule(sched)

  # with LoggingRank(f'batch_id: {batch_id}, finish _exec_schedule barrier'):
  #   dist.barrier()
  # print_rank_0(f'fist stage end _exec_schedule, batch_id: {batch_id}')
  # 返回采样结果
  if mpu.is_pipeline_last_stage():
    # 只有最后一个stage 有output
    logits = model.outputs
    assert logits is not None
    logits = logits[0, :, -1].view(1, -1).contiguous()  # (vocab_size,)
    logits = logits.float()
    logits /= args.temperature
    logits = top_k_logits(logits, top_k=args.top_k, top_p=args.top_p)

    log_probs = F.softmax(logits, dim=-1)
    new_token_id = torch.multinomial(log_probs, num_samples=1).view(-1).long()

    src = mpu.get_pipeline_model_parallel_last_rank()
    assert src == dist.get_rank()
    # 将数据发送到第一层
    group = mpu.get_embedding_group()
    # new_token_id = torch.LongTensor([1]).cuda()
    with LoggingRank(
        f'batch_id: {batch_id}, last stage broadcast new_token_id'):
      torch.distributed.broadcast(new_token_id, src, group)

    # with LoggingRank(
    #     f'batch_id: {batch_id}, last stage get_embedding_group barrier'):
    #   dist.barrier(group)
    # print(
    #     'last stage',
    #     f'batch_id: {batch_id}, rank:{dist.get_rank()}:group: {torch.distributed.get_world_size(group=group)},src:{src},rank in embed:{torch.distributed.get_rank(group=group)}',
    #     f'new_token_id: {new_token_id},{type(new_token_id)}'
    #     )
  elif mpu.is_pipeline_first_stage():
    # print(f'first stage enter elif: {batch_id}')
    # 第一个stage中接受消息
    src = mpu.get_pipeline_model_parallel_last_rank()
    group = mpu.get_embedding_group()
    # print(f'first stage get src group:  {batch_id}')
    # new_token_id = torch.empty_like(torch.LongTensor([1])).cuda()
    # global dump_tensor
    # if dump_tensor is None:
    new_token_id = torch.LongTensor([0]).cuda()
    # new_token_id = dump_tensor
    # print('fist stage start broadcast',f'batch_id: {batch_id},pid: {os.getpid()}')
    with LoggingRank(
        f'batch_id: {batch_id}, first stage broadcast new_token_id'):
      torch.distributed.broadcast(new_token_id, src, group)

    # with LoggingRank(
    #     f'batch_id: {batch_id}, first stage get_embedding_group barrier'):
    #   dist.barrier(group)
    # print(
    #     'first stage',
    #     f'batch_id: {batch_id}',
    #     new_token_id,
    #     new_token_id.shape,
    #     f'rank:{dist.get_rank()},group:{group},src:{src},rank in embed:{torch.distributed.get_rank(group=group)}',
    #     f'new_token_id: {new_token_id}'
    #     )
    # print()
  else:
    new_token_id = None

  dist.barrier()

  # print(f'rank {dist.get_rank()} 到达barrieer')
  # group = mpu.get_embedding_group()
  # dist.barrier()
  # print(f'rank {dist.get_rank()} 通过barrieer')
  # with LoggingRank(f'batch_id: {batch_id}, last barrier'):
  #   dist.barrier()
  return new_token_id


def is_pipeline_first_or_last_stage():
  return mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage()


def generate_samples_ds(model, prompt: str, max_gen_length):
  """
    使用deepspeed进行采样
    promt
    """
  assert max_gen_length > 0, max_gen_length
  args = get_args()
  tokenizer = get_tokenizer()
  print_rank_0(f'prompt: {prompt}')
  prompt_tokens_ids = tokenizer.tokenize(prompt)
  ori_prompt_len = torch.LongTensor([len(prompt_tokens_ids)]).cuda()
  prompt_tokens_ids = torch.LongTensor(prompt_tokens_ids).cuda()
  # global  dump_tensor
  # dump_tensor = torch.LongTensor([0]).cuda()

  print_rank_0(
      f'ori prompt_tokens_ids:{prompt_tokens_ids},ori_prompt_len:{ori_prompt_len}'
  )
  from megatron import print_with_rank

  # dist.barrier()
  with torch.no_grad():
    for i in range(max_gen_length):
      #
      # dist.barrier()
      # print(f'rank {dist.get_rank()}, i:{i}')
      # 保证每一个进程的序列长度都一样
   
      torch.distributed.broadcast(
        ori_prompt_len, 
        mpu.get_pipeline_model_parallel_last_rank(),
        mpu.get_pipeline_model_parallel_group()
        )
      
      print_with_rank('ori_prompt_len', ori_prompt_len,f'i: {i}')

      if is_pipeline_first_or_last_stage():
        data_iter = iter([{'text': prompt_tokens_ids}])
      else:
        data_iter = iter([{
            'text':
            torch.LongTensor([1] * ori_prompt_len[0].item())
        }])

      # with LoggingRank(f'batch_id: {i}, generate_one_token_ds'):
      new_token_id = generate_one_token_ds(model, data_iter, i)
      # print(f'rank return {dist.get_rank()}, i:{i}')
      if is_pipeline_first_or_last_stage():
        # print(f'rank {dist.get_rank()}, cat tensor: {i}')
        prompt_tokens_ids = torch.cat((prompt_tokens_ids, new_token_id),
                                      dim=-1)
        ori_prompt_len[0] = prompt_tokens_ids.shape[0]

      # print(f'rank {dist.get_rank()}, cat tensor: {i},prompt len: {ori_prompt_len[0].item()}')
      # dist.barrier()
      #
  print_rank_0(f'gener prompt_tokens_ids:{prompt_tokens_ids}')

  # 将结果发送到所有进程中

  prompt_tokens_ids_len = torch.LongTensor([prompt_tokens_ids.shape[0]]).cuda()
  torch.distributed.broadcast(prompt_tokens_ids_len, 0)

  if not is_pipeline_first_or_last_stage():
    prompt_tokens_ids = torch.LongTensor(
        [0] * prompt_tokens_ids_len[0].item()).cuda()

  torch.distributed.broadcast(prompt_tokens_ids, 0)
  # 做detokenize
  decode_tokens = tokenizer.detokenize(
      prompt_tokens_ids.cpu().numpy().tolist())

  return decode_tokens[ori_prompt_len:]


def generate_samples_interactive(
  model, 
  max_new_tokens, 
  tokenizer,
  eos_token_id,
  max_seq_len,
  recompute=False,
  greedy=False,
  temperature=1,
  top_k=0,
  top_p=1,
  multiturn=True
  ):

  args = get_args()
  tokenizer = get_tokenizer()

  context_count = 0
  model.eval()
  os.system('clear')
  historys = []
  with torch.no_grad():
    while True:
      terminate_runs = 0
      raw_text_len = 0
      
      if mpu.is_pipeline_first_stage() \
         and mpu.get_tensor_model_parallel_rank() == 0:
        # os.system('clear')
        raw_text = input("\nContext prompt (stop to exit) >>> ")
        while not raw_text:
          print('Prompt should not be empty!')
          raw_text = input("\nContext prompt (stop to exit) >>> ")
        
        raw_input = raw_text
        if multiturn:
          history_str = '</s>'.join([history[0] + history[1] for history in historys])
          if historys:
            history_str += '</s>'

          history_str += raw_text
          raw_text = history_str
          # print('\n raw_text: ',raw_text)


        # raw_text = f"Humman:{raw_text}\nAssistant:"
        raw_text_len = len(raw_text)
        raw_text_wrap = prompt_template(raw_text)
        if "stop" in raw_text:
          terminate_runs = 1
        else:
          context_tokens = tokenizer.tokenize(raw_text_wrap)
          context_length = len(context_tokens)

          if context_length >= (args.seq_length // 2) :
            print("\nContext length",
                  context_length, "\nPlease give smaller context (half of the "
                  "sequence length)!",
                  flush=True)
            continue
          if raw_input == "clear":
            historys = []
            continue
            
      else:
        context_tokens = tokenizer.tokenize("EMPTY TEXT")
        context_length = 0

      input_info = [terminate_runs, raw_text_len, context_length]
      input_info_tensor = torch.cuda.LongTensor(input_info)
      torch.distributed.all_reduce(input_info_tensor,
                                   group=mpu.get_model_parallel_group())
      terminate_runs = input_info_tensor[0].item()
      raw_text_len = input_info_tensor[1].item()
      context_length = input_info_tensor[2].item()

      if terminate_runs == 1:
        return

      # For pipeline parallel we send context tokens to other stages
      # so they get the lengths correct
      if mpu.get_tensor_model_parallel_rank() == 0 \
         and args.pipeline_model_parallel_size > 1:
        if mpu.is_pipeline_first_stage():
          src = mpu.get_pipeline_model_parallel_first_rank()
          group = mpu.get_pipeline_model_parallel_group()
          context_tokens_tensor = torch.cuda.LongTensor(context_tokens)
          torch.distributed.broadcast(context_tokens_tensor, src, group)
        else:
          src = mpu.get_pipeline_model_parallel_first_rank()
          group = mpu.get_pipeline_model_parallel_group()
          context_tokens_tensor = torch.empty(context_length,
                                              dtype=torch.int64,
                                              device=torch.device("cuda"))
          torch.distributed.broadcast(context_tokens_tensor, src, group)
          context_tokens = context_tokens_tensor.cpu().numpy().tolist()

      token_stream = get_token_stream(
        model, 
        [context_tokens],
        max_generated_len=min(
          max_new_tokens+len(context_tokens),max_seq_len
          ),
        max_seq_len=max_seq_len, 
        eos_token_id=eos_token_id,
        recompute=recompute,
        greedy=greedy,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p
        )
      
      if mpu.is_pipeline_first_stage() \
         and mpu.get_tensor_model_parallel_rank() == 0:
        print("\nContext:", raw_input, flush=True)

        print("\nAssistant:",flush=True,end='')
        last_str =  raw_text_wrap        

      for counter, decode_tokens in enumerate(token_stream):
        # time.sleep(0.1)
        if mpu.is_pipeline_first_stage() \
         and mpu.get_tensor_model_parallel_rank() == 0:
        # os.system('clear')
          if not isinstance(decode_tokens, list):
            decode_tokens, _ = decode_tokens
            decode_tokens = decode_tokens[0].cpu().numpy().tolist()
          trim_decode_tokens = tokenizer.detokenize(decode_tokens[:-1])[raw_text_len:]
          trim_decode_tokens_str = "".join(trim_decode_tokens)
          last_str_ = trim_decode_tokens_str
          trim_decode_tokens_str = trim_decode_tokens_str.lstrip(last_str)
          last_str = last_str_

          print(f"{trim_decode_tokens_str}", flush=True,end='')
      
      if mpu.is_pipeline_first_stage() \
         and mpu.get_tensor_model_parallel_rank() == 0:
        if multiturn:
          historys.append((raw_input,trim_decode_tokens))
        # if counter % print_frequency != 0 \
        #    or mpu.get_tensor_model_parallel_rank() != 0 \
        #    or not mpu.is_pipeline_first_stage():
        #   continue

        # os.system('clear')
        # print("\nContext:", raw_text, flush=True)

        # decode_tokens, _ = decode_tokens
        # decode_tokens = decode_tokens[0].cpu().numpy().tolist()
        # trim_decode_tokens = tokenizer.detokenize(decode_tokens)[raw_text_len:]
        # print("\nAssistant:", trim_decode_tokens, flush=True)

      # if mpu.is_pipeline_first_stage() \
      #    and mpu.get_tensor_model_parallel_rank() == 0:
      #   # os.system('clear')
      #   print("\nContext:", raw_input, flush=True)

      #   if not isinstance(decode_tokens, list):
      #     decode_tokens, _ = decode_tokens
      #     decode_tokens = decode_tokens[0].cpu().numpy().tolist()
      #   trim_decode_tokens = tokenizer.detokenize(decode_tokens[:-1])[raw_text_len:]
      #   print("\nAssistant:", trim_decode_tokens, flush=True)
      #   if multiturn:
      #     historys.append((raw_input,trim_decode_tokens))
        # input("\nPress Enter to continue >>>")

      raw_text = None
      context_count += 1


def distribute_tokenize(prompts,tokenizer,max_len):
  """
  使用分布式的方式进行tokenizer,然后进行reduce。
  每一个rank都需要进行计算

  Args:
      prompts (_type_): _description_
  """
  
  current_rank = dist.get_rank()
  world_size = dist.get_world_size()
  prompt_mum_per_rank = math.ceil(len(prompts)/world_size)
  current_rank_prompts = prompts[current_rank*prompt_mum_per_rank:(current_rank+1)*prompt_mum_per_rank]
  current_rank_list = []
  
  if current_rank_prompts:
    current_rank_list = [tokenizer.tokenize(current_rank_prompt)[:max_len] for current_rank_prompt in current_rank_prompts]
  
  output = [None]*world_size
  dist.all_gather_object(output, current_rank_list)
  output = sum(output,[])
  return output
  

def generate_samples(
    model, 
    prompts, 
    max_new_tokens, 
    tokenizer,
    eos_token_id,
    max_seq_len,
    recompute=False,
    greedy=False,
    temperature=1,
    top_k=0,
    top_p=1,
    batch_size=None
    ):
  """_summary_

  Args:
      model (_type_): _description_
      prompts (_type_): _description_
      batch_size (int, optional): _description_. Defaults to 2.

  Returns:
      _type_: _description_
  """
  # 支持以batch的形式进行解码

  args = get_args()
  if batch_size is None:
    batch_size = args.micro_batch_size
  tokenizer = get_tokenizer()
  if isinstance(prompts,str):
    prompts = [prompts]
  
  assert len(prompts) >0, prompts
  raw_text_lens = [len(prompt) for prompt in prompts]

  # 分布式tokenizer
  tokenized_prompts = distribute_tokenize(prompts,tokenizer,args.seq_length)
  assert len(tokenized_prompts) == len(prompts)

  # if dist.get_rank()==0:
  #   print(len(prompts))
  #   for prompt in tokenized_prompts:
  #     print(prompt)
  
  # batch_num = math.ceil(len(prompts)/batch_size)
  generates = []
  model.eval()
  batch_starts = list(range(0,len(tokenized_prompts),batch_size))
  if dist.get_rank() == 0:
    batch_starts = tqdm(batch_starts)

  with torch.no_grad():
    for batch_start in batch_starts:
      torch.cuda.empty_cache()
      tokenized_prompt_batch = tokenized_prompts[batch_start:batch_start+batch_size]
      raw_text_len_batch = raw_text_lens[batch_start:batch_start+batch_size]
      # tokenizer, 仅仅对于张量并行的第一个位置做,然后不同到其他机器
      contexts_tensor_len = [len(context_tokens) for context_tokens in tokenized_prompt_batch]
      max_context_tensor_len = max(contexts_tensor_len)
      # print_rank_0('max_context_tensor_len',max_context_tensor_len,max_new_tokens,max_seq_len)

      token_stream = get_token_stream(
        model,tokenized_prompt_batch,
        max_generated_len=min(
          max_new_tokens+max_context_tensor_len,max_seq_len
          ),
        max_seq_len=max_seq_len, 
        eos_token_id=eos_token_id,
        recompute=recompute,
        greedy=greedy,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p
        )

      decode_tokens = None
      for counter, decode_tokens in enumerate(token_stream):
        # print('counter',counter)
        continue
     
      decode_tokens, lengths = decode_tokens
      
      # decode_tokens = decode_tokens[0].tolist()
      if mpu.is_pipeline_last_stage() \
        and mpu.get_tensor_model_parallel_rank() == 0:
        if decode_tokens is None:
          trim_decode_token_list = [None]*len(tokenized_prompt_batch)
        else:
          trim_decode_token_list = [
            tokenizer.detokenize(decode_tokens[i,:lengths[i]])[raw_text_len_batch[i]:]
            for i in range(len(tokenized_prompt_batch))
          ]

        # 将数据发送到 rank 0
        torch.distributed.broadcast_object_list(
          trim_decode_token_list,
          torch.distributed.get_rank(),
          mpu.get_embedding_group()
          )
        generates.extend(trim_decode_token_list)
      elif mpu.is_pipeline_first_stage() \
        and mpu.get_tensor_model_parallel_rank() == 0:
        trim_decode_token_list = [None]*len(tokenized_prompt_batch)
        torch.distributed.broadcast_object_list(
          trim_decode_token_list,
          mpu.get_pipeline_model_parallel_last_rank(),
          mpu.get_embedding_group()
          )

        generates.extend(trim_decode_token_list)
        # print(trim_decode_token_list)
      
    return generates

def generate_samples_stream(
    model, 
    prompts, 
    max_new_tokens, 
    tokenizer,
    eos_token_id,
    max_seq_len,
    recompute=False,
    greedy=False,
    temperature=1,
    top_k=0,
    top_p=1,
    batch_size=None
    ):
  """_summary_

  Args:
      model (_type_): _description_
      prompts (_type_): _description_
      batch_size (int, optional): _description_. Defaults to 2.

  Returns:
      _type_: _description_
  """
  # 支持以batch的形式进行解码
  
  args = get_args()
  if batch_size is None:
    batch_size = args.micro_batch_size
  tokenizer = get_tokenizer()
  if isinstance(prompts,str):
    prompts = [prompts]
  assert len(prompts) == 1, 'stream model only support bs 1'
  
  assert len(prompts) >0, prompts
  raw_text_lens = [len(prompt) for prompt in prompts]

  # 分布式tokenizer
  tokenized_prompts = distribute_tokenize(prompts,tokenizer,args.seq_length)
  assert len(tokenized_prompts) == len(prompts)

  # if dist.get_rank()==0:
  #   print(len(prompts))
  #   for prompt in tokenized_prompts:
  #     print(prompt)
  
  # batch_num = math.ceil(len(prompts)/batch_size)
  # generates = []
  model.eval()
  with torch.no_grad():
    for batch_start in tqdm(range(0,len(tokenized_prompts),batch_size),total=len(tokenized_prompts)//batch_size):
      torch.cuda.empty_cache()
      tokenized_prompt_batch = tokenized_prompts[batch_start:batch_start+batch_size]
      raw_text_len_batch = raw_text_lens[batch_start:batch_start+batch_size]
      # tokenizer, 仅仅对于张量并行的第一个位置做,然后不同到其他机器
      contexts_tensor_len = [len(context_tokens) for context_tokens in tokenized_prompt_batch]
      max_context_tensor_len = max(contexts_tensor_len)
      # print_rank_0('max_context_tensor_len',max_context_tensor_len,max_new_tokens,max_seq_len)

      token_stream = get_token_stream(
        model,tokenized_prompt_batch,
        max_generated_len=min(
          max_new_tokens+max_context_tensor_len,max_seq_len
          ),
        max_seq_len=max_seq_len, 
        eos_token_id=eos_token_id,
        recompute=recompute,
        greedy=greedy,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p
        )

      # decode_tokens = None
      for counter, decode_tokens in enumerate(token_stream):
        # print_with_rank(counter)
        decode_tokens, lengths = decode_tokens
        # decode_tokens = decode_tokens[0].tolist()
        if mpu.is_pipeline_last_stage() \
          and mpu.get_tensor_model_parallel_rank() == 0:
          # print('get new token')
          if decode_tokens is None:
            trim_decode_token_list = [None]*len(tokenized_prompt_batch)
          else:
            # 将最后一个位置的id返回出去
            trim_decode_token_list = [
              decode_tokens[i,-1].tolist()
              for i in range(len(tokenized_prompt_batch))
            ]

          # 将数据发送到 rank 0
          torch.distributed.broadcast_object_list(
            trim_decode_token_list,
            torch.distributed.get_rank(),
            mpu.get_embedding_group()
            )
          yield trim_decode_token_list
          # generates.extend(trim_decode_token_list)
        elif mpu.is_pipeline_first_stage() \
          and mpu.get_tensor_model_parallel_rank() == 0:
          trim_decode_token_list = [None]*len(tokenized_prompt_batch)
          torch.distributed.broadcast_object_list(
            trim_decode_token_list,
            mpu.get_pipeline_model_parallel_last_rank(),
            mpu.get_embedding_group()
            )
          
          yield trim_decode_token_list
        else:
          yield []

          # generates.extend(trim_decode_token_list)
        # print(trim_decode_token_list)
      
    # return generates
# def get_batch_with_pipeline(context_tokens):
#   """Generate batch from context tokens."""
#   args = get_args()
#   tokenizer = get_tokenizer()

#   # Move to GPU.
#   # tokens = context_tokens.view(args.micro_batch_size, -1).contiguous().cuda()
#   tokens = context_tokens.contiguous().cuda()
#   # Get the attention mask and position ids.
#   attention_mask, _, position_ids = get_ltor_masks_and_position_ids(
#       tokens,
#       tokenizer.eod,
#       False,
#       False,
#       args.eod_mask_loss,
#       prefix_indices=None,
#       loss_on_targets_only=args.loss_on_targets_only)
#   # attention_mask = None
#   return tokens, attention_mask, position_ids


def generate_samples_with_pipeline(
    model, 
    prompts, 
    max_new_tokens, 
    tokenizer,
    eos_token_id,
    max_seq_len,
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
      context_length = task['context_length']
      logits = task['logits']
      tokens = task['input_ids']
      lengths = task['lengths']
      if greedy:
        prev = torch.argmax(logits, dim=-1).view(-1)
      else:
        logits = logits.float()
        #TODO 加入random温度函数
        logits /= temperature
        logits = top_k_logits(logits, top_k=top_k, top_p=top_p)
        log_probs = F.softmax(logits, dim=-1)
        # print('log_probs.shape',log_probs.shape)
        prev = torch.multinomial(log_probs, num_samples=1).view(-1)
      started = (task['context_lengths'] <= task['context_length']).cuda()
      # print(started,tokens,prev)
      new_tokens = switch(tokens[:, context_length].view(-1).cuda(), prev, started)
      # 将新的token写入到input_ids中
      if context_length > tokens.shape[1]:
        # 此时需要进行将新的token拼接到后面
        tokens = torch.cat((tokens,new_tokens.unsqueeze(-1)),dim=-1)
      else:
        tokens[:, context_length] = new_tokens
      task['input_ids'] = tokens
     
      # 将new token发送到first stage中
      src = mpu.get_pipeline_model_parallel_last_rank()
      group = mpu.get_embedding_group()
      torch.distributed.broadcast(new_tokens, src, group)
      # print_with_rank('last broadcast new tokens')
      # 判断当前任务是否结束,并将结束符发送到其他rank
      is_done = task['done_tensor']
      done_token = (prev == eos_token_id).byte() & started.byte()
      just_finished = (done_token & ~is_done).bool()
      lengths[just_finished.view(-1)] = context_length
      is_done = is_done | done_token

      task['done_tensor'] = is_done
    
      done = torch.all(is_done)
      src = mpu.get_pipeline_model_parallel_last_rank()
      group = mpu.get_pipeline_model_parallel_group()
      torch.distributed.broadcast(done, src, group)
      task['is_done'] = done
      
    else:
      if mpu.is_pipeline_first_stage():
        context_length = task['context_length']
        tokens = task['input_ids']
        src = mpu.get_pipeline_model_parallel_last_rank()
        group = mpu.get_embedding_group()
        new_tokens = torch.empty_like(tokens[:, context_length]).cuda()
        torch.distributed.broadcast(new_tokens, src, group)
        # print_with_rank('first broadcast new tokens')
        # 将新的token写入到input_ids中
        if context_length > tokens.shape[1]:
          # 此时需要进行将新的token拼接到后面
          tokens = torch.cat((tokens,new_tokens.unsqueeze(-1)),dim=-1)
        else:
          tokens[:, context_length] = new_tokens
        task['input_ids'] = tokens
      # print_with_rank('other broadcast done')
      done = torch.cuda.ByteTensor([0])
      src = mpu.get_pipeline_model_parallel_last_rank()
      group = mpu.get_pipeline_model_parallel_group()
      torch.distributed.broadcast(done, src, group)
      task['is_done'] = done
    
    task['context_length'] += 1
    # 此时超过了序列的最大长度，需要退出
    if task['context_length']  >= task["max_seq_len"]:
      task['is_done'] = True
      

    # if task['is_done']:
    #    print_rank_0(f'task {task["task_id"]} done')


  args = get_args()
  args.variable_seq_lengths = True
  tokenizer = get_tokenizer()
  if isinstance(prompts,str):
    prompts = [prompts]
  
  assert len(prompts) >0, prompts
  raw_text_lens = [len(prompt) for prompt in prompts]

  # 分布式tokenizer
  tokenized_prompts = distribute_tokenize(prompts,tokenizer,args.seq_length)
  assert len(tokenized_prompts) == len(prompts)

  # pipeline_batches = []
  # 任务构造，数据暂时不加载到gpu, 减小显存
  tasks_queue = queue.PriorityQueue()
  for i,batch_start in enumerate(range(0,len(prompts),micro_batch_size)):
    prompts_batch = prompts[batch_start:batch_start+micro_batch_size]
    tokenized_prompts_batch = tokenized_prompts[batch_start:batch_start+micro_batch_size]
    raw_text_lens_batch = raw_text_lens[batch_start:batch_start+micro_batch_size]
    # context_lengths = [len(prompt) for prompt in prompts_batch]
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
    max_seq_len_ = min(
        max_new_tokens+context_length,max_seq_len)
    task = {
        "prompts_batch":prompts_batch,
        "input_ids":input_ids,
        "position_ids": position_ids,
        "attention_mask": attention_mask,
        "task_id": i,
        "tokentype_ids":None,
        "layer_past": None,
        "context_length": context_length,
        "context_lengths":context_lengths,
        "new_token_num":0,
        "raw_text_lens":raw_text_lens_batch,
        "max_seq_len": max_seq_len_,
        "done_tensor": torch.zeros([input_ids.shape[0]]).byte().cuda(), 
        "is_done": False, 
        "logits": None,
        "lengths": torch.ones([input_ids.shape[0]]).long() * max_seq_len_
      }
    
    # print_rank_0('task id',task['task_id'],task['context_length'])
    tasks_queue.put((i,task))
  
  # # 实例化任务队列
  # tasks_queue = queue.PriorityQueue()
  ret_queue = queue.PriorityQueue()
  # print(sfdg)
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
      forward_step_with_pipelining(
        model,
        tasks,
        forward_method_parallel_output=False
      )
      # for task in tasks:
      #   if task['task_id'] == 4:
      #     print_rank_last('task id',task['task_id'],task['prompts_batch'],task['context_length'],task['lengths'])
      
      # 生成采样, 没有结束的任务继续送到任务队列中
      # TODO
      for task in tasks:
        # print_with_rank("sampling...")
        _sampling(task)
        if task['is_done']:
          # 当前任务结束,进行后处理
          torch.cuda.empty_cache()
          ret_queue.put(
            (task['task_id'],task)
          )
          if dist.get_rank() == 0:
            pbar.update(1)
        else:
          tasks_queue.put((task['task_id'],task))
    # 结果后处理并传到first_stage 里面去
    generates = []
    while not ret_queue.empty():
      task = ret_queue.get()[1]
      if mpu.is_pipeline_last_stage() \
        and mpu.get_tensor_model_parallel_rank() == 0:

        # if task['task_id'] == 4:
        #   print(task)

        decode_tokens = task['input_ids']
        lengths = task['lengths']
        raw_text_lens = task['raw_text_lens']
        trim_decode_token_list = [
            tokenizer.detokenize(decode_tokens[i,:lengths[i]])[raw_text_lens[i]:]
            for i in range(decode_tokens.shape[0])
          ]
        trim_decode_token_list = [
          {
            "trim_decode_token":trim_decode_token,
            "task_id":task['task_id']
          }
          for trim_decode_token in trim_decode_token_list
          ]
        # 将数据发送到 rank 0
        torch.distributed.broadcast_object_list(
          trim_decode_token_list,
          torch.distributed.get_rank(),
          mpu.get_embedding_group()
          )
        generates.extend(
          [d['trim_decode_token'] for d in trim_decode_token_list]
          )

      elif mpu.is_pipeline_first_stage() \
        and mpu.get_tensor_model_parallel_rank() == 0:
        decode_tokens = task['input_ids']
        trim_decode_token_list = [None]*decode_tokens.shape[0]
        torch.distributed.broadcast_object_list(
          trim_decode_token_list,
          mpu.get_pipeline_model_parallel_last_rank(),
          mpu.get_embedding_group()
        )
        for d in trim_decode_token_list:
          assert task['task_id'] == d['task_id'], (task['task_id'],d['task_id'])
          generates.extend(
            [d['trim_decode_token']]
          )
        # generates.extend(trim_decode_token_list)
  
  return generates
      
     

def forward_step_with_pipelining(
  model,
  tasks,
  forward_method_parallel_output=False
  ):
  """
  pipeline forward 算法的实现
  """
  # Compute number of warmup microbatches.
  # batch_size,maxlen = tokens.shape
  # assert batch_size % micro_batch_size == 0, (batch_size,micro_batch_size)
  
  # num_microbatches = batch_size % micro_batch_size
  # num_warmup_microbatches = \
  #     (mpu.get_pipeline_model_parallel_world_size() -
  #      mpu.get_pipeline_model_parallel_rank() - 1)
  # num_warmup_microbatches = min(num_warmup_microbatches, num_microbatches)
  # num_microbatches_remaining = \
  #     num_microbatches - num_warmup_microbatches
  # args = get_args()
  # orig_seq_length = args.seq_length
  # args.seq_length = tokens.shape[1]
  # input_tensors = []
  # output_tensors = []
  # losses_reduced = []
  unwrapped_model = unwrap_model(model, (torchDDP, LocalDDP, Float16Module))

  # 结果写入
  # logits_list = []
  # layer_past_list = []
  # if mpu.is_pipeline_last_stage():
  #   logits = torch.empty(
  #       (batch_size, maxlen, padded_vocab_size),
  #       dtype=torch.float16, device=torch.cuda.current_device())

  # Run warmup forward passes.
  for task in tasks:
    input_tensor = recv_forward()
    unwrapped_model.set_input_tensor(input_tensor)
    # print(task)
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
    # if layer_past is not None:
    #   layer_past_use = (layer_past[0][:,start:end,...],layer_past[1][:,start:end,...])
    # else:
    #   layer_past_use = None
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
    # layer_past_list.append(layer_past)
    
    send_forward(output_tensor)
    if mpu.is_pipeline_last_stage():
      # logits 计算
      logits = output_tensor[:, -1].contiguous()
      task['logits'] = logits
  

  # dist.barrier()



def generate_samples_unconditional(model):

  args = get_args()
  tokenizer = get_tokenizer()

  num_samples = args.num_samples
  context_tokens = [[tokenizer.eod] for _ in range(args.micro_batch_size)]
  ctr = 0
  while True:
    start_time = time.time()
    for token_stream in get_token_stream(model, copy.deepcopy(context_tokens)):
      pass
    if mpu.is_pipeline_last_stage() and \
       mpu.get_tensor_model_parallel_rank() == 0:
      if ctr % args.log_interval == 0:
        print('Avg s/batch:',
              (time.time() - start_time) / min(args.log_interval, ctr + 1))
        start_time = time.time()
      length = len(token_stream)
      token_batch = token_stream[0].cpu().numpy().tolist()
      length_batch = token_stream[1].cpu().numpy().tolist()
      assert len(length_batch) == args.micro_batch_size
      for tokens, length in zip(token_batch, length_batch):
        tokens = tokens[1:length - 1]
        text = tokenizer.detokenize(tokens)
        is_finished = length < args.seq_length - 1
        datum = {'text': text, 'length': length - 1, 'finished': is_finished}
        yield datum
        ctr += 1
        if ctr >= num_samples:
          break
    else:
      for _ in range(args.micro_batch_size):
        yield None
        ctr += 1
        if ctr >= num_samples:
          break
    if ctr >= num_samples:
      break


def generate_and_write_samples_unconditional(model):

  args = get_args()
  assert args.genfile is not None
  with open(args.genfile, 'w') as f:
    for datum in generate_samples_unconditional(model):
      if mpu.is_pipeline_last_stage() and \
         mpu.get_tensor_model_parallel_rank() == 0:
        f.write(json.dumps(datum) + '\n')


def pad_batch(batch, pad_id, args):

  context_lengths = []
  for tokens in batch:
    context_length = len(tokens)
    if context_length < args.seq_length:
      tokens.extend([pad_id] * (args.seq_length - context_length))
    context_lengths.append(context_length)
  return batch, context_lengths


def get_token_stream(model, context_tokens, 
    max_generated_len,
    eos_token_id,
    max_seq_len,
    recompute=False,
    greedy=False,
    temperature=1,
    top_k=0,
    top_p=1,
    ):

  args = get_args()
  tokenizer = get_tokenizer()

  context_tokens, context_lengths = pad_batch(context_tokens, tokenizer.eod,
                                              args)
  

  # print(f'padding 之后的context: {context_tokens}')

  context_tokens_tensor = torch.cuda.LongTensor(context_tokens)
  context_length_tensor = torch.cuda.LongTensor(context_lengths)

  torch.distributed.broadcast(context_length_tensor,
                              mpu.get_tensor_model_parallel_src_rank(),
                              group=mpu.get_tensor_model_parallel_group())
  torch.distributed.broadcast(context_tokens_tensor,
                              mpu.get_tensor_model_parallel_src_rank(),
                              group=mpu.get_tensor_model_parallel_group())

  context_length = context_length_tensor.min().item()
  tokens, attention_mask, position_ids = get_batch(context_tokens_tensor)

  batch_token_iterator = sample_sequence_batch(model, context_tokens_tensor,
                                               context_length_tensor,
                                               attention_mask, position_ids,
                                               eos_token_id,
                                               max_generated_len,
                                               recompute=recompute,
                                               greedy=greedy,
                                               temperature=temperature,
                                               top_k=top_k,
                                               top_p=top_p,
                                               )
  for tokens, lengths in batch_token_iterator:
    context_length += 1
    if tokens is not None:
      yield tokens[:, :context_length], lengths
    else:
      yield None, None


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
  # print_with_rank('wait recive tensor')
  input_tensor = recv_forward()
  # print_with_rank('get recive tensor')

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
  # print_with_rank('sending next tensor')
  send_forward(output_tensor)
  # print_with_rank('sent next tensor')

  args.seq_length = orig_seq_length
  if get_key_value:
    return output_tensor, layer_past
  return output_tensor



def __forward_step_with_pipelining(
  model,
  micro_batch_size,
  tokens,
  position_ids,
  attention_mask,
  tokentype_ids,
  layer_past=None,
  get_key_value=None,
  forward_method_parallel_output=None
  ):
  """Run non-interleaved 1F1B schedule, with communication between pipeline
    stages.

    Returns dictionary with losses if the last stage, empty dict otherwise.
  """
  # Compute number of warmup microbatches.
  batch_size,maxlen = tokens.shape
  assert batch_size % micro_batch_size == 0, (batch_size,micro_batch_size)
  
  num_microbatches = batch_size % micro_batch_size
  # num_warmup_microbatches = \
  #     (mpu.get_pipeline_model_parallel_world_size() -
  #      mpu.get_pipeline_model_parallel_rank() - 1)
  # num_warmup_microbatches = min(num_warmup_microbatches, num_microbatches)
  # num_microbatches_remaining = \
  #     num_microbatches - num_warmup_microbatches
  args = get_args()
  # orig_seq_length = args.seq_length
  # args.seq_length = tokens.shape[1]
  # input_tensors = []
  # output_tensors = []
  # losses_reduced = []
  unwrapped_model = unwrap_model(model, (torchDDP, LocalDDP, Float16Module))

  # 结果写入
  logits_list = []
  layer_past_list = []
  # if mpu.is_pipeline_last_stage():
  #   logits = torch.empty(
  #       (batch_size, maxlen, padded_vocab_size),
  #       dtype=torch.float16, device=torch.cuda.current_device())

  # Run warmup forward passes.
  for micro_batch_index in range(len(num_microbatches)):
    start = micro_batch_index * micro_batch_size
    end = min(start + micro_batch_size, batch_size)
    input_tensor = p2p_communication.recv_forward()
    unwrapped_model.set_input_tensor(input_tensor)

    tokens_use = tokens[start:end,...]
    position_ids_use = position_ids[start:end,...]
    attention_mask_use = attention_mask[start:end,...]
    tokentype_ids_use = tokentype_ids[start:end,...]
    if layer_past is not None:
      layer_past_use = (layer_past[0][:,start:end,...],layer_past[1][:,start:end,...])
    else:
      layer_past_use = None
    
    output_tensor = unwrapped_model(
      tokens_use, 
      position_ids_use,
      attention_mask_use,
      tokentype_ids=tokentype_ids_use,
      layer_past=layer_past_use,
      forward_method_parallel_output=forward_method_parallel_output
      )
    
    if get_key_value:
      output_tensor, layer_past = output_tensor
      layer_past_list.append(layer_past)
    
    p2p_communication.send_forward(output_tensor)
    if mpu.is_pipeline_last_stage():
      logits_list.append(output_tensor)
  
  # args.seq_length = orig_seq_length

  # 结果融合
  logits = None
  if mpu.is_pipeline_last_stage():
    logits = torch.cat(logits_list,dim=0)
  
  if get_key_value:
    layer_past_0 = torch.cat([i[0] for i in layer_past_list],dim=1)
    layer_past_1 = torch.cat([i[1] for i in layer_past_list],dim=1)
    layer_past = (layer_past_0,layer_past_1)
    
  # 结果写入
  if get_key_value:
    return logits,layer_past
  else:
    return logits


def sample_sequence_batch(model,
                          context_tokens,
                          context_lengths,
                          attention_mask,
                          position_ids,
                          eos_id,
                          maxlen,
                          greedy=False,
                          temperature=1,
                          top_k=0,
                          top_p=1,
                          type_ids=None,
                          recompute=False
                          ):

  # args = get_args()
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
    org_context_length = context_length

    layer_past = None
    batch_size = context_tokens.size(0)
    is_done = torch.zeros([batch_size]).byte().cuda()
    tokens = context_tokens
    # if maxlen is None:
    #   maxlen = args.seq_length - 1
    #   if maxlen > (org_context_length + args.out_seq_length):
    #     maxlen = org_context_length + args.out_seq_length

    lengths = torch.ones([batch_size]).long().cuda() * maxlen
    # print_rank_0('context_length',context_length,maxlen,tokens.shape,position_ids.shape)
    while context_length < (maxlen):
      # print_rank_0(f'run : context_length: {context_length}')
      if recompute:
        output = forward_step(model,
                              tokens,
                              position_ids,
                              attention_mask,
                              tokentype_ids=type_ids,
                              forward_method_parallel_output=False)
        if mpu.is_pipeline_last_stage():
          assert output is not None
          logits = output[:, context_length - 1, :]
      else:
        types2use = None
        if counter == 0:
          tokens2use = tokens[:, :context_length]
          positions2use = position_ids[:, :context_length]
          if type_ids is not None:
            types2use = type_ids[:, :context_length]
        else:
          tokens2use = tokens[:, context_length - 1].view(batch_size, -1)
          try:
            positions2use = position_ids[:, context_length - 1].view(
                batch_size, -1)
          except :
            print(tokens.shape,position_ids.shape,context_length,maxlen)
            raise 
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
        if mpu.is_pipeline_last_stage():
          assert output is not None
          logits = output[:, -1].view(batch_size, -1).contiguous()

      if mpu.is_pipeline_last_stage():
        if greedy:
          prev = torch.argmax(logits, dim=-1).view(-1)
        else:
          logits = logits.float()
          #TODO 加入random温度函数
          logits /= temperature
          logits = top_k_logits(logits, top_k=top_k, top_p=top_p)
          log_probs = F.softmax(logits, dim=-1)
          prev = torch.multinomial(log_probs, num_samples=1).view(-1)

        started = context_lengths <= context_length

        new_tokens = switch(tokens[:, context_length].view(-1), prev, started)
        tokens[:, context_length] = new_tokens
        src = mpu.get_pipeline_model_parallel_last_rank()
        group = mpu.get_embedding_group()
        # print_with_rank('send new tokens')
        torch.distributed.broadcast(new_tokens, src, group)
        
        

        done_token = (prev == eos_id).byte() & started.byte()
        just_finished = (done_token & ~is_done).bool()
        lengths[just_finished.view(-1)] = context_length
        is_done = is_done | done_token

        yield tokens, lengths 
        # print_with_rank('send done ')
        done = torch.all(is_done)
        src = mpu.get_pipeline_model_parallel_last_rank()
        group = mpu.get_pipeline_model_parallel_group()
        torch.distributed.broadcast(done, src, group)
        
      else:
        if mpu.is_pipeline_first_stage():
          src = mpu.get_pipeline_model_parallel_last_rank()
          group = mpu.get_embedding_group()
          new_tokens = torch.empty_like(tokens[:, context_length])
          torch.distributed.broadcast(new_tokens, src, group)
          tokens[:, context_length] = new_tokens
          # print_with_rank('get new tokens')
          yield tokens, None
        else:
          yield None, None

        done = torch.cuda.ByteTensor([0])
        src = mpu.get_pipeline_model_parallel_last_rank()
        group = mpu.get_pipeline_model_parallel_group()
        torch.distributed.broadcast(done, src, group)

        # print_with_rank('get done ')

      context_length += 1
      counter += 1
      if done:
        break

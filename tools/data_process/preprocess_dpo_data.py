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
"""
dpo训练的数据构造。
这里事先将引用模型logrpob计算出来。

"""
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from megatron.data.indexed_dataset import best_fitting_dtype
from megatron.model.gpt_model_ppo import GPTModelWithPPOValueHead
from multiprocessing.pool import ThreadPool
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import json
import multiprocessing
import os
import pickle
import sys
import threading
import time
import torch

logprobs_of_labels = GPTModelWithPPOValueHead.logprobs_of_labels

try:
  import nltk
  nltk_available = True
except ImportError:
  nltk_available = False

from megatron.tokenizer import build_tokenizer
from megatron.data import indexed_dataset

thread_local_data = threading.local()


# https://stackoverflow.com/questions/33139531/preserve-empty-lines-with-nltks-punkt-tokenizer
class CustomLanguageVars(nltk.tokenize.punkt.PunktLanguageVars):

  _period_context_fmt = r"""
        \S*                          # some word material
        %(SentEndChars)s             # a potential sentence ending
        \s*                       #  <-- THIS is what I changed
        (?=(?P<after_tok>
            %(NonWord)s              # either other punctuation
            |
            (?P<next_tok>\S+)     #  <-- Normally you would have \s+ here
        ))"""


class IdentitySplitter(object):
  def tokenize(self, *text):
    return text


class Encoder(object):
  def __init__(self, args):
    self.args = args

  def initializer(self):
    # Use Encoder class as a container for global data
    Encoder.tokenizer = build_tokenizer(self.args)

  def encode(self, json_line):
    # out data process method
    data = eval(json_line)
    prompt = data['conversations'][0]['value']
    chosen = data['chosen']
    rejected = data['rejected']

    prompt_ids = Encoder.tokenizer.tokenize(prompt)
    chosen_ids = Encoder.tokenizer.tokenize(chosen)
    rejected_ids = Encoder.tokenizer.tokenize(rejected)

    if not self.args.no_append_eod:
      chosen_ids.append(Encoder.tokenizer.eod)
      rejected_ids.append(Encoder.tokenizer.eod)

    # # 去掉太长的序列, 包括太长的prompt以及太长的回答
    if len(prompt_ids) + len(chosen_ids) > self.args.ref_seq_length or \
      len(prompt_ids) + len(rejected_ids) > self.args.ref_seq_length or \
        len(prompt_ids) >= self.args.ref_seq_length:       \
            return {}

    return {
        'prompt_ids': prompt_ids,
        'chosen_ids': chosen_ids,
        'rejected_ids': rejected_ids
    }


def get_args():
  parser = argparse.ArgumentParser()
  group = parser.add_argument_group(title='input data')
  group.add_argument('--input', type=str, help='Path to input JSON')

  group.add_argument('--output', type=str, default='./', help='Path to output')

  group = parser.add_argument_group(title='tokenizer')
  group.add_argument('--tokenizer-type',
                     type=str,
                     required=True,
                     choices=[
                         'BertWordPieceLowerCase', 'BertWordPieceCase',
                         'GPT2BPETokenizer', 'PretrainedFromHF'
                     ],
                     help='What type of tokenizer to use.')
  group.add_argument('--vocab-file',
                     type=str,
                     default=None,
                     help='Path to the vocab file')
  group.add_argument('--merge-file',
                     type=str,
                     default=None,
                     help='Path to the BPE merge file (if necessary).')

  group.add_argument("--tokenizer-name-or-path",
                     type=str,
                     default=None,
                     help="Name or path of the huggingface tokenizer.")
  group.add_argument('--make-vocab-size-divisible-by',
                     type=int,
                     default=128,
                     help='Pad the vocab size to be divisible by this value.'
                     'This is added for computational efficieny reasons.')

  group.add_argument(
      '--pad-vocab-size-to',
      type=int,
      default=None,
      help='Pad the vocab size to be divisible by this value.'
      'Value of the size of the vocabulary of the tokenizer to reach. This value must be greater than'
      ' the initial size of the tokenizer. If this argument is used the value of '
      '`make-vocab-size-divisible-by` will be ignored.')

  group = parser.add_argument_group(title='output data')

  group = parser.add_argument_group(title='runtime')

  group.add_argument('--no-append-eod',
                     action='store_true',
                     help='append eod to chosen or reject')
  group.add_argument('--workers',
                     type=int,
                     default=1,
                     help='Number of worker processes to launch')
  group.add_argument('--log-interval',
                     type=int,
                     default=100,
                     help='Interval between progress updates')

  group.add_argument('--ref-batch-size',
                     type=int,
                     default=4,
                     help='batch of refrence model')

  group.add_argument('--ref-seq-length',
                     type=int,
                     default=2048,
                     help='seq-length of refrence model')

  group.add_argument('--ref-model-name', type=str, help='refrence model name')

  group.add_argument('--ref-model-devices',
                     type=lambda x: x.split(','),
                     default='0',
                     help='for using multi gpu, set `0,1,2,3`')

  args = parser.parse_args()

  # some default/dummy values for the tokenizer
  args.rank = 0
  args.tensor_model_parallel_size = 1
  args.vocab_extra_ids = 0

  return args


# def init_model(args):
#   # global TOKENIZER
#   model_name = args.model_name
#   # tokenizer = AutoTokenizer.from_pretrained(model_name)
#   model = AutoModelForCausalLM.from_pretrained(model_name,torch_dtype=torch.float16).cuda()
#   # tokenizer.padding_side = 'right'
#   # TOKENIZER = tokenizer
#   return model


def __model_call(args, post_data, model: torch.nn.Module):
  query_lens = [
      len(query_tensor) for query_tensor in post_data['query_tensors']
  ]
  respond_lens = [
      len(response_tensor) for response_tensor in post_data['response_tensors']
  ]
  device = next(model.parameters()).device
  input_ids_list = [
      torch.LongTensor(query_tensor + response_tensor).to(device)
      for query_tensor, response_tensor in zip(post_data['query_tensors'],
                                               post_data['response_tensors'])
  ]
  # ser_input,temperature,top_p,max_gen_len=25
  input_ids = pad_sequence(input_ids_list, batch_first=True,
                           padding_value=0).to(device)
  input_ids = input_ids[:, :args.ref_seq_length]

  with torch.no_grad():
    output = model(input_ids=input_ids, return_dict=True)
    logits = output['logits']
    logprobs_list = []
    for i in range(logits.shape[0]):
      q_len = query_lens[i]
      res_len = respond_lens[i]
      # cur_seq_len = q_len + res_len
      start = q_len - 1
      end = res_len + start
      # 去掉太长的句子
      assert q_len + res_len <= args.ref_seq_length, (q_len, res_len)

      # end = start + respond_lens_batch[i]
      logprobs_list.append([
          logprobs_of_labels(logits[i, start:end, :],
                             input_ids[i, q_len:q_len + res_len]).tolist()
      ])
  return logprobs_list


def batch_iterator(encoded_feats, batch_size):
  batch = []
  for encoded_feat in encoded_feats:
    batch.append(encoded_feat)
    if len(batch) == batch_size:
      yield batch
      batch = []

  if batch:
    yield batch


def tokenize_stage(args, chunk_size=25):
  fin = open(args.input, 'r', encoding='utf-8')
  encoder = Encoder(args)
  pool = multiprocessing.Pool(args.workers, initializer=encoder.initializer)
  encode_feats_iterator = pool.imap(encoder.encode, fin, chunk_size)
  for encode_feat in encode_feats_iterator:
    if len(encode_feat) > 0:
      yield encode_feat


MODEL = None


def ref_model_initialize(args):
  global MODEL
  process_index = multiprocessing.current_process()._identity[0]

  def _get_device():

    return args.ref_model_devices[process_index % len(args.ref_model_devices)]

  # model_attr = 'ref_model'
  if MODEL is None:
    model_name = args.ref_model_name
    print(f'processing: {process_index}, geting model device')
    device = _get_device()
    print(
        f'processing: {process_index}, initializing model on device: {device}')
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16).to(f'cuda:{device}')
    print(
        f'processing: {process_index}, initialized model on device: {device}')
    MODEL = model
    Encoder.tokenizer = build_tokenizer(args)


def ref_model_call(args, batch_data):
  model = MODEL
  prompt_batch = [b['prompt_ids'] for b in batch_data]
  chosen_batch = [b['chosen_ids'] for b in batch_data]
  rejected_batch = [b['rejected_ids'] for b in batch_data]

  prompt_lens = [len(i) for i in prompt_batch]
  chosen_lens = [len(i) for i in chosen_batch]
  rejected_lens = [len(i) for i in rejected_batch]

  device = next(model.parameters()).device

  chosen_ids_list = []
  rejected_ids_list = []
  for prompt_ids, chosen_ids, rejected_ids in zip(prompt_batch, chosen_batch,
                                                  rejected_batch):
    chosen_ids_list.append(
        torch.LongTensor(prompt_ids + chosen_ids).to(device))
    rejected_ids_list.append(
        torch.LongTensor(prompt_ids + rejected_ids).to(device))

  input_ids_list = chosen_ids_list + rejected_ids_list

  # # prompt_lens = [len(query_tensor) for query_tensor in batch_data]
  # # respond_lens = [len(response_tensor) for response_tensor in post_data['response_tensors']]
  # device = next(model.parameters()).device
  # input_ids_list = [
  #   torch.LongTensor(query_tensor+response_tensor).to(device)
  #   for query_tensor,response_tensor in zip(post_data['query_tensors'],post_data['response_tensors'])
  #   ]

  # 注意这里pad的值需要与训练过程保持一致
  input_ids = pad_sequence(input_ids_list,
                           batch_first=True,
                           padding_value=Encoder.tokenizer.pad).to(device)
  # input_ids = input_ids[:,:args.ref_seq_length]

  with torch.no_grad():
    output = model(input_ids=input_ids, return_dict=True)
    logits = output['logits']
    bs = logits.shape[0] // 2

    chosen_logits = logits[:bs, :, :]
    rejected_logits = logits[bs:, :, :]

    chosen_ids = input_ids[:bs, :]
    rejected_ids = input_ids[bs:, :]

    # logprobs_list = []
    for i in range(bs):
      chosen_len = chosen_lens[i]
      rejected_len = rejected_lens[i]

      max_chosen_rejected_len = max(chosen_len, rejected_len)

      chosen_logit = chosen_logits[i, prompt_lens[i] - 1:prompt_lens[i] - 1 +
                                   max_chosen_rejected_len]
      rejected_logit = rejected_logits[i, prompt_lens[i] - 1:prompt_lens[i] -
                                       1 + max_chosen_rejected_len]

      chosen_id = chosen_ids[i][prompt_lens[i]:prompt_lens[i] +
                                max_chosen_rejected_len]
      rejected_id = rejected_ids[i][prompt_lens[i]:prompt_lens[i] +
                                    max_chosen_rejected_len]

      chosen_logprob = logprobs_of_labels(chosen_logit, chosen_id)

      rejected_logprob = logprobs_of_labels(rejected_logit, rejected_id)

      batch_data[i]['chosen_logprob'] = chosen_logprob
      batch_data[i]['rejected_logprob'] = rejected_logprob

    return batch_data

  #     q_len = query_lens[i]
  #     res_len = respond_lens[i]
  #     # cur_seq_len = q_len + res_len
  #     start = q_len -1
  #     end = res_len + start
  #     # 去掉太长的句子
  #     assert q_len + res_len <= args.ref_seq_length,(q_len,res_len)

  #     # end = start + respond_lens_batch[i]
  #     logprobs_list.append([
  #       logprobs_of_labels(
  #         logits[i, start:end,:],
  #         input_ids[i, q_len:q_len + res_len]
  #       ).tolist()
  #     ])
  # return logprobs_list


# def ref_model_heler(args,batch_data):
#   # 线程名称
#   # thread_name = threading.currentThread().getName()
#   # model_attr = 'ref_model'
#   # if not hasattr(thread_local_data,model_attr):
#   #   model_name = args.ref_model_name
#   #   # tokenizer = AutoTokenizer.from_pretrained(model_name)
#   #   model = AutoModelForCausalLM.from_pretrained(model_name,torch_dtype=torch.float16).to(f'cuda:{device}')
#   #   setattr(args,model_attr,model)
#   model = MODEL
#   # print(type(batch_data))
#   # print(sfg)

#   # prompt_batch = [b['prompt_ids'] for b in batch_data]
#   # chosen_batch = [b['chosen_ids'] for b in batch_data]
#   # rejected_batch =[b['rejected_ids'] for b in batch_data]
#   # chosen_post_data = {
#   #   "query_tensors": prompt_batch,
#   #   "response_tensors": chosen_batch
#   # }
#   # rejected_post_data = {
#   #   "query_tensors": prompt_batch,
#   #   "response_tensors": rejected_batch
#   # }

#   model_call(args,batch_data,model)

#   # rejected_logprobs = model_call(args,rejected_post_data,model)
#   # # assert len(rejected_logprobs) == len(batch_data)
#   # # assert len(chosen_logprobs) == len(batch_data)
#   # ret = []
#   # for i,b in enumerate(batch_data):
#   #   b['chosen_logprob'] = chosen_logprobs[i]
#   #   b['rejected_logprob'] = rejected_logprobs[i]
#   #   # 去掉空的元素
#   #   if b['chosen_logprob'] and b['rejected_logprob']:
#       # ret.append(b)

#   return batch_data


def ref_model_stage(args, feats_iterator):
  """
  调用模型进行计算,使用多线程支持多个gpu的计算
  Args:
      feats_iterator (_type_): _description_
      ref_model (_type_): _description_
      ref_batch_size (_type_): _description_
  """

  devices = args.ref_model_devices
  worker_num = len(devices)
  #
  # device_get_lock = multiprocessing.Lock()
  pool = multiprocessing.Pool(worker_num,
                              initializer=ref_model_initialize,
                              initargs=(args, ))
  iters = pool.imap(partial(ref_model_call, args),
                    batch_iterator(feats_iterator, args.ref_batch_size))
  for iter_ in iters:
    yield from iter(iter_)


def save_stage(args, res_iterator):
  """
  写入函数
  Args:
      res_iterator (_type_): _description_
      save_path (_type_): _description_
  """
  ret = []
  for i in res_iterator:
    ret.append(i)
    yield

  save_path = os.path.join(
      args.output +
      f'{os.path.basename(args.input)}.feat.sample_num.{len(ret)}.pkl')
  print(f'save to: ', save_path)
  with open(save_path, 'wb') as f:
    pickle.dump(ret, f)


def run_pipeline(stages, args, total_sample_num=None):
  # args.total_sample_num = total_sample_num

  assert len(stages) > 0, stages
  last_iter = None
  for i, stage in enumerate(stages):
    if i == 0:
      last_iter = stage(args)
    else:
      last_iter = stage(args, last_iter)

  for _ in tqdm(last_iter, total=total_sample_num):
    pass


def main():
  args = get_args()
  startup_start = time.time()
  print("Opening", args.input)

  stages = [tokenize_stage, ref_model_stage, save_stage]

  run_pipeline(stages, args, 200000 // args.ref_batch_size)
  print(f'execute time: {time.time()-startup_start}')


if __name__ == '__main__':
  torch.multiprocessing.set_start_method('spawn')
  main()
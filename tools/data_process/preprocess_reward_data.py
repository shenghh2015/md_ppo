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
"""reward模型数据预处理"""

import argparse
import json
import multiprocessing
import os
import sys
import pickle
from megatron import prompt_template
from megatron.data.indexed_dataset import best_fitting_dtype
import time

import torch
try:
  import nltk
  nltk_available = True
except ImportError:
  nltk_available = False

from megatron.tokenizer import build_tokenizer
from megatron.data import indexed_dataset


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
    prompt = data['prompt']
    # prompt = prompt_template(data['prompt'])
    chosen = data['chosen']
    rejected = data['rejected']
    prompt_ids = Encoder.tokenizer.tokenize(prompt)
    chosen_ids = Encoder.tokenizer.tokenize(chosen)
    rejected_ids = Encoder.tokenizer.tokenize(rejected)

    return {
        'prompt_ids': prompt_ids,
        'chosen_ids': chosen_ids,
        'rejected_ids': rejected_ids
    }


def get_args():
  parser = argparse.ArgumentParser()
  group = parser.add_argument_group(title='input data')
  group.add_argument('--input', type=str, help='Path to input JSON')

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
  group.add_argument('--append-eod',
                     action='store_true',
                     help='Append an <eod> token to the end of a document.')
  group.add_argument('--append-ans',
                     action='store_true',
                     help='Append an <ans> token to the end of a prefix.')
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
  group.add_argument('--workers',
                     type=int,
                     default=1,
                     help='Number of worker processes to launch')
  group.add_argument('--log-interval',
                     type=int,
                     default=100,
                     help='Interval between progress updates')
  args = parser.parse_args()

  # some default/dummy values for the tokenizer
  args.rank = 0
  args.tensor_model_parallel_size = 1
  args.vocab_extra_ids = 0

  return args


def main():
  args = get_args()
  startup_start = time.time()

  print("Opening", args.input)

  fin = open(args.input, 'r', encoding='utf-8')
  encoder = Encoder(args)
  tokenizer = build_tokenizer(args)
  pool = multiprocessing.Pool(args.workers, initializer=encoder.initializer)
  encoded_feats = list(pool.imap(encoder.encode, fin, 25))

  #encoded_docs = map(encoder.encode, fin)
  output_path = args.input + f'.feat.sample_num.{len(encoded_feats)}.pkl'
  with open(output_path, 'wb') as f:
    pickle.dump(encoded_feats, f)

  print(f'save to {output_path},execute time: {time.time()-startup_start}')


if __name__ == '__main__':
  main()
  # python tools/preprocess_my_data.py --input data/webqa/webqa.json --output-prefix data/webqa/webqa_only_eod  --vocab config/gpt2/gpt2-vocab.json --dataset-impl mmap --tokenizer-type GPT2BPETokenizer --merge-file config/gpt2/gpt2-merges.txt --append-eod --workers 8 --append-ans
  # python tools/preprocess_sft_data.py --input chinese_data/alpaca_chatgpt/chinese_alpaca_data.json --output-prefix chinese_data/alpaca_chatgpt/trans_chinese_alpaca_data  --dataset-impl mmap --tokenizer-type PretrainedFromHF  --append-eod --workers 8 --append-ans --tokenizer-name-or-path bloom-add-ans-tokenizer

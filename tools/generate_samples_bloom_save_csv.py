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
"""Sample Generate BLOOM"""

import os
import sys
import torch
import random
from megatron import get_args
from megatron import print_rank_0
from megatron import get_tokenizer
from megatron import mpu
from megatron.checkpointing import load_checkpoint
from megatron.initialize import initialize_megatron
from megatron.model.gpt_model import GPTModel
from megatron.training import get_model
from megatron.text_generation_utils import generate_and_write_samples_unconditional
from megatron.text_generation_utils import generate_samples_input_from_file
from megatron.text_generation_utils import generate_samples_interactive
from megatron.text_generation_utils import generate_samples

from megatron.enums import AttnMaskType
from megatron.model import GPTModel, GPTModelPipe
import deepspeed
import pandas as pd


def model_provider(args, pre_process=True, post_process=True):
  """Build the model."""

  print_rank_0('building GPT model ...')
  # GPTModel.set_input_tensor
  model = GPTModel(num_tokentypes=0,
                   parallel_output=False,
                   pre_process=pre_process,
                   post_process=post_process)

  # args.pretrain_causal_attention = True
  # model = GPTModelPipe(
  #     num_tokentypes=0,
  #     parallel_output=True,
  #     attn_mask_type=AttnMaskType.causal
  # )

  # print_rank_0("DeepSpeed is enabled.")
  #     #pp = mpu.get_pipeline_model_parallel_world_size()

  # import json
  # import io
  # with io.open(args.deepspeed_config, "r", encoding="utf-8") as f:
  #     config = json.load(f)
  # if args.universal_checkpoint:
  #     config["checkpoint"] = {"load_universal": True}

  # model, optimizer, _, lr_scheduler = deepspeed.initialize(
  #         model=model,
  #         optimizer=None,
  #         lr_scheduler=None,
  #         config=config,
  #         args=args,
  #     )

  return model


def prompts_provider(path):
  import csv
  with open(path, 'r') as f:
    raw_data = csv.reader(f)
    lines = [i for i in raw_data][1:]
    prompts = [i[0] for i in lines]
    dates = [i[-2] for i in lines]
    return prompts, dates


def add_text_generate_args(parser):
  """Text generation arguments."""
  group = parser.add_argument_group(title='text generation')

  group.add_argument("--temperature",
                     type=float,
                     default=0.7,
                     help='Sampling temperature.')
  group.add_argument("--greedy",
                     action='store_true',
                     default=False,
                     help='Use greedy sampling.')
  group.add_argument("--top_p",
                     type=float,
                     default=0.0,
                     help='Top p sampling.')
  group.add_argument("--top_k", type=int, default=0, help='Top k sampling.')
  group.add_argument("--out-seq-length",
                     type=int,
                     default=1024,
                     help='Size of the output generated text.')
  group.add_argument("--sample-input-file",
                     type=str,
                     default=None,
                     help='Get input from file instead of interactive mode, '
                     'each line is an input.')
  group.add_argument("--sample-output-file",
                     type=str,
                     default=None,
                     help='Output file got from --sample-input-file')
  group.add_argument("--num-samples",
                     type=int,
                     default=0,
                     help='Number of samples to generate unconditionally, '
                     'defaults to 0 and interactive conditional sampling')
  group.add_argument("--genfile",
                     type=str,
                     help='Output file when generating unconditionally')
  group.add_argument("--recompute",
                     action='store_true',
                     help='During generation recompute all attention '
                     'instead of using previously computed keys/values.')

  return parser


def main():
  """Main program."""

  initialize_megatron(extra_args_provider=add_text_generate_args,
                      args_defaults={
                          'tokenizer_type': 'GPT2BPETokenizer',
                          'no_load_rng': True,
                          'no_load_optim': True
                      })

  args = get_args()
  if args.num_layers_per_virtual_pipeline_stage is not None:
    print(
        "Interleaved pipeline schedule is not yet supported for text generation."
    )
    exit()

  # Set up model and load checkpoint.
  from functools import partial
  model = get_model(partial(model_provider, args))

  if args.load is not None:
    _ = load_checkpoint(model, None, None)

  assert len(model) == 1, "Above condition should have caught this"
  model = model[0]

  # with open('chinese_data/belle_eval/selected_eval_set.json','r') as f:
  #   lines = f.readlines()
  #   lines = list(map(eval, lines))
  # prompts = [i['question'] for i in lines]

  # prompts = [
  #     '个人去银行办理小额贷款，需要哪些条件和流程',
  #     '申请公积金贷款的流程是怎样的',
  #     '我在简单贷借款3####一年还清总利息是多少还要服务费吗',
  # ]
  path = 'chinese_data/annotated_data/流行度高_重要度中_3210_n-best.csv'
  prompts, dates = prompts_provider(path)
  prompts = random.sample(prompts, 100)

  for index, prompt in enumerate(prompts):
    new_prompt = f'Human:{prompt} \nAssitant:'
    single_prompts = [new_prompt] * 7
    output_list = []
    for i in single_prompts:
      args.temperature = random.uniform(0.5, 0.7)
      print_rank_0(f'tem is {args.temperature}')
      output = generate_samples(model, i)
      print_rank_0(f'prompt: {prompt}, output:{output}')
      if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
          output_list.append(output.lstrip(' ？').strip('</s>'))
    if torch.distributed.is_initialized():
      if torch.distributed.get_rank() == 0:
        output_log = pd.DataFrame([[prompt, output_list[0], output_list[1], output_list[2], output_list[3], \
                                    output_list[4], output_list[5], output_list[6], dates[index]]],
                                  columns=['prompt', 'answer1', 'answer2', 'answer3', 'answer4', \
                                           'answer5', 'answer6', 'answer7', 'date'])
        output_log.to_csv(
            'generated_output/bloomz-7b1-mt_custom_random-temperature_output_log.csv',
            mode='a',
            header=False,
            index=False)
  # Generate samples.
  # if args.num_samples == 0:
  #     args.micro_batch_size = 1
  #     if args.sample_input_file != None:
  #         generate_samples_input_from_file(model)
  #     else:
  #         generate_samples_interactive(model)
  # else:
  #     generate_and_write_samples_unconditional(model)


if __name__ == "__main__":
  main()

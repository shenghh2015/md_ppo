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
import random
import pandas as pd
from tqdm import tqdm
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
from tools.generate_samples_bloom_base import add_text_generate_args
import deepspeed


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


def sequence_clean(output):
  return output.lstrip(' ？').strip('</s>')


def random_temperature(low, high):
  return random.uniform(low, high)


# def add_text_generate_args(parser):
#   """Text generation arguments."""
#   group = parser.add_argument_group(title='text generation')

#   group.add_argument("--temperature",
#                      type=float,
#                      default=0.7,
#                      help='Sampling temperature.')
#   group.add_argument("--greedy",
#                      action='store_true',
#                      default=False,
#                      help='Use greedy sampling.')
#   group.add_argument("--top_p",
#                      type=float,
#                      default=0.0,
#                      help='Top p sampling.')
#   group.add_argument("--top_k", type=int, default=0, help='Top k sampling.')
#   group.add_argument("--out-seq-length",
#                      type=int,
#                      default=1024,
#                      help='Size of the output generated text.')
#   group.add_argument("--sample-input-file",
#                      type=str,
#                      default=None,
#                      help='Get input from file instead of interactive mode, '
#                      'each line is an input.')
#   group.add_argument("--sample-output-file",
#                      type=str,
#                      default=None,
#                      help='Output file got from --sample-input-file')
#   group.add_argument("--num-samples",
#                      type=int,
#                      default=0,
#                      help='Number of samples to generate unconditionally, '
#                      'defaults to 0 and interactive conditional sampling')
#   group.add_argument("--genfile",
#                      type=str,
#                      help='Output file when generating unconditionally')
#   group.add_argument("--recompute",
#                      action='store_true',
#                      help='During generation recompute all attention '
#                      'instead of using previously computed keys/values.')

#   return parser


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

  # prompts = [
  #     '我是一个公司员工，想请假休息三天，帮我写个请假条',
  #     '你好',
  #     '你的开发者是谁',
  #     '用长120厘米的铁丝围成一个长方形，长是宽的1.5倍，求它的宽是多少厘米?',
  # ]
  with open('chinese_data/belle_2m/random_10k_prompts.json', 'r') as f:
    lines = f.readlines()
    lines = list(map(eval, lines))
  raw_prompts = [i['instruction'] for i in lines]

  temperatures = [0.7, 1, 1.125]
  # temperatures = random_temperature(0.5, 0.7)
  prompts = [f'Human:{i}\nAssitant:' for i in raw_prompts]
  multi_prompts = prompts * 5
  random.shuffle(multi_prompts)

  ids_prompt_dict = dict(zip(range(len(multi_prompts)), multi_prompts))
  prompt_ids_dict = {}
  for key in ids_prompt_dict:
    value = ids_prompt_dict[key]
    prompt_ids_dict.setdefault(value, []).append(key)

  import time
  start = time.time()
  tokenizer = get_tokenizer()
  outputs = generate_samples(model,
                             multi_prompts,
                             args.max_new_tokens,
                             tokenizer,
                             tokenizer.eod,
                             args.seq_length,
                             recompute=args.recompute,
                             temperature=temperatures,
                             top_p=args.top_p,
                             top_k=args.top_k,
                             greedy=args.greedy,
                             batch_size=args.micro_batch_size)

  for i, prompt in enumerate(prompts):
    # print_rank_0(f'prompt: {raw_prompts[i]},')
    # print_rank_0(f'output: {output}')
    answer_ids = prompt_ids_dict[prompt]
    output_list = []
    for ids in answer_ids:
      output_list.append(outputs[ids])
    print_rank_0(f'output: {output_list}')
    # import torch
    # if torch.distributed.is_initialized():
    #   if torch.distributed.get_rank() == 0:
    #     filter_output = list(map(sequence_clean, output_list))
    #     output_log = pd.DataFrame(
    #       [[raw_prompts[i], filter_output[0], filter_output[1], filter_output[2], filter_output[3], filter_output[4]]],
    #       columns=['prompt','answer1', 'answer2', 'answer3', 'answer4', 'answer5'])
    #     output_log.to_csv('generated_output/bloomz-7b1-mt_10k_output.csv', mode='a', header=False, index=False)

  end = time.time()
  print_rank_0(f'whole process costs :{end-start}s')


if __name__ == "__main__":
  main()

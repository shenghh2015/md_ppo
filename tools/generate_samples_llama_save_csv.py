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
import pandas as pd
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
from deepspeed.runtime.utils import see_memory_usage
from llama.llama_model import ModelArgs, Attention, FeedForward, TransformerBlock, Transformer, precompute_freqs_cis


def model_provider(pre_process=True, post_process=True):
  """Build the model."""

  print_rank_0('building GPT model ...')
  if torch.distributed.is_initialized(
  ) and not torch.distributed.get_rank() == 0:
    pass
  see_memory_usage(f"Before Building Model", force=True)

  args = get_args()
  print(args)
  # norm_eps is set to 1e-5 only for 65B models
  if args.num_layers == 80:
    norm_eps = 1e-5
  else:
    norm_eps = 1e-6
  model_args = ModelArgs(dim=args.hidden_size,
                         n_layers=args.num_layers,
                         n_heads=args.num_attention_heads,
                         norm_eps=norm_eps,
                         max_seq_len=args.max_position_embeddings)

  freqs_cis = precompute_freqs_cis(model_args.dim // model_args.n_heads,
                                   model_args.max_seq_len)

  with deepspeed.zero.Init(data_parallel_group=mpu.get_data_parallel_group(),
                           remote_device=None if args.remote_device == 'none'
                           else args.remote_device,
                           config_dict_or_path=args.deepspeed_config,
                           enabled=args.zero_stage == 3,
                           mpu=mpu):
    if args.deepspeed:
      args.pretrain_causal_attention = True
      model = GPTModelPipe(num_tokentypes=0,
                           parallel_output=True,
                           attn_mask_type=AttnMaskType.causal)
      # This is a hack to give us a reference to get_batch_pipe from within training.py
      # We need to call model.set_batch_fn after deepspeed.initialize
      model._megatron_batch_fn = get_batch_pipe
    else:
      model = Transformer(model_args,
                          pre_process=pre_process,
                          post_process=post_process)

  for k, v in model.state_dict().items():
    print(k, v.shape)
  device = next(model.parameters()).device
  # 这一步看似多余但是是必须的，因为nn.parameter()注册的变量没有被移动到相应的device上
  model = model.to(device)

  see_memory_usage(f"After Building Model", force=True)

  # raise NotImplementedError
  return model


# def model_provider(args, pre_process=True, post_process=True):
#   """Build the model."""

#   print_rank_0('building GPT model ...')
#   # GPTModel.set_input_tensor
#   model = GPTModel(num_tokentypes=0,
#                    parallel_output=False,
#                    pre_process=pre_process,
#                    post_process=post_process)

#   # args.pretrain_causal_attention = True
#   # model = GPTModelPipe(
#   #     num_tokentypes=0,
#   #     parallel_output=True,
#   #     attn_mask_type=AttnMaskType.causal
#   # )

#   # print_rank_0("DeepSpeed is enabled.")
#   #     #pp = mpu.get_pipeline_model_parallel_world_size()

#   # import json
#   # import io
#   # with io.open(args.deepspeed_config, "r", encoding="utf-8") as f:
#   #     config = json.load(f)
#   # if args.universal_checkpoint:
#   #     config["checkpoint"] = {"load_universal": True}

#   # model, optimizer, _, lr_scheduler = deepspeed.initialize(
#   #         model=model,
#   #         optimizer=None,
#   #         lr_scheduler=None,
#   #         config=config,
#   #         args=args,
#   #     )

#   return model


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
  # model = get_model(partial(model_provider, args))
  model = get_model(model_provider)

  if args.load is not None:
    _ = load_checkpoint(model, None, None)

  assert len(model) == 1, "Above condition should have caught this"
  model = model[0]
  with open('chinese_data/belle_eval/selected_eval_set.json', 'r') as f:
    lines = f.readlines()
    lines = list(map(eval, lines))
  prompts = [i['question'] for i in lines]

  # prompts = [
  #     '中国的首都是哪座城市?',
  #     '海南省的省会是哪座城市?',
  # ]
  # 可能会出现None
  for index, prompt in enumerate(prompts):
    new_prompt = f'Human:{prompt} \nAssitant:'
    single_prompts = [new_prompt] * 4
    output_list = []
    for i in single_prompts:
      output = generate_samples(model, i)
      print_rank_0(f'prompt: {prompt}, output:{output}')
      if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
          output_list.append(output)
    if torch.distributed.is_initialized():
      if torch.distributed.get_rank() == 0:
        output_log = pd.DataFrame([[
            prompt, output_list[0], output_list[1], output_list[2],
            output_list[3], lines[index]['class'], lines[index]['std_answer']
        ]],
                                  columns=[
                                      'prompt', 'answer1', 'answer2',
                                      'answer3', 'answer4', 'class',
                                      'std_answer'
                                  ])
        output_log.to_csv(
            'generated_output/llama-7b_custom_wo_cpu-offload_output_log2.csv',
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

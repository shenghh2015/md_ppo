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
import os
import sys
from datetime import timedelta
from megatron import get_args
from megatron import print_rank_0
from megatron import get_tokenizer
import torch.distributed as dist
from megatron import mpu
from megatron.checkpointing import load_checkpoint
from megatron.initialize import initialize_megatron
from megatron.model.gpt_model import GPTModel
from megatron.training import get_model
from megatron.text_generation_utils import generate_and_write_samples_unconditional
from megatron.text_generation_utils import generate_samples_input_from_file
from megatron.text_generation_utils import generate_samples_interactive
from megatron.text_generation_utils import generate_samples
from megatron.text_generation_utils import generate_samples_interactive
from megatron.text_generation_utils import generate_samples_ds

from megatron.enums import AttnMaskType
from megatron.model import GPTModel, GPTModelPipe
import deepspeed
import torch


def get_batch_pipe(data):
  # args = get_args()
  # tokenizer = get_tokenizer()
  # print_rank_0(data)
  # Items and their type.
  keys = ['text']
  datatype = torch.int64

  # 将输入数据同步到其他张量并行中
  data_b = mpu.broadcast_data(keys, data, datatype)
  # print_rank_0('finish broadcast data')

  # Unpack.
  tokens = data_b['text'].long()
  # print('get_batch',tokens.shape)
  tokens = tokens.view(1, -1).contiguous().cuda()
  micro_batch_size, seq_length = tokens.size()

  attention_mask = torch.tril(
      torch.ones((1, seq_length, seq_length),
                 device=tokens.device)).view(1, 1, seq_length, seq_length)

  position_ids = torch.arange(seq_length,
                              dtype=torch.long,
                              device=tokens.device)
  position_ids = position_ids.unsqueeze(0).expand_as(tokens)
  loss_mask = torch.ones(tokens.size(),
                         dtype=torch.float,
                         device=tokens.device)

  # labels = tokens_[:, 1:].contiguous()
  # tokens = tokens_[:, :-1].contiguous()

  # assert data_b.shape[0] == 1,

  # # Get the masks and position ids.
  # attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
  #     tokens,
  #     tokenizer.eod,
  #     args.reset_position_ids,
  #     args.reset_attention_mask,
  #     args.eod_mask_loss,
  #     prefix_indices=None,
  #     loss_on_targets_only=args.loss_on_targets_only
  # )
  # if args.curriculum_learning and args.curriculum_seqlen < tokens.size()[1]:
  #     # seqlen-based curriculum learning
  #     # tokens, position_ids, labels, loss_mask have size [batch size, seqlen]
  #     tokens = tokens[:, :args.curriculum_seqlen].contiguous()
  #     position_ids = position_ids[:, :args.curriculum_seqlen].contiguous()
  #     labels = labels[:, :args.curriculum_seqlen].contiguous()
  #     loss_mask = loss_mask[:, :args.curriculum_seqlen].contiguous()

  return (tokens, position_ids, attention_mask), (tokens, loss_mask)


def model_provider(pre_process=True, post_process=True):
  """Build the model."""

  # print_rank_0('building GPT model ...')
  # model = GPTModel(num_tokentypes=0, parallel_output=False,
  #                  pre_process=pre_process, post_process=post_process)

  args = get_args()
  args.pretrain_causal_attention = True
  # 实例化模型
  model = GPTModelPipe(num_tokentypes=0,
                       parallel_output=False,
                       attn_mask_type=AttnMaskType.causal)
  # This is a hack to give us a reference to get_batch_pipe from within training.py
  # We need to call model.set_batch_fn after deepspeed.initialize
  # model._megatron_batch_fn = get_batch_pipe

  # 实例化deepspeed engine
  print_rank_0("DeepSpeed is enabled.")
  import json
  import io
  with io.open(args.deepspeed_config, "r", encoding="utf-8") as f:
    config = json.load(f)
  if args.universal_checkpoint:
    config["checkpoint"] = {"load_universal": True}
  model, optimizer, _, lr_scheduler = deepspeed.initialize(
      model=model,
      optimizer=None,
      lr_scheduler=None,
      config=config,
      args=args,
  )
  # get_batch_pipe 用来对数据进行梳理，get_batch_pipe的返回值就直接进行函数执行
  model.set_batch_fn(get_batch_pipe)

  assert model.grid.get_pipe_parallel_rank(
  ) == mpu.get_pipeline_model_parallel_rank()
  assert model.grid.get_slice_parallel_rank(
  ) == mpu.get_tensor_model_parallel_rank()
  assert model.grid.get_data_parallel_rank() == mpu.get_data_parallel_rank()

  return model


def add_text_generate_args(parser):
  """Text generation arguments."""
  group = parser.add_argument_group(title='text generation')

  group.add_argument("--temperature",
                     type=float,
                     default=1.0,
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
  # group.add_argument("--out-seq-length", type=int, default=1024,
  #                    help='Size of the output generated text.')
  # group.add_argument("--sample-input-file", type=str, default=None,
  #                    help='Get input from file instead of interactive mode, '
  #                    'each line is an input.')
  # group.add_argument("--sample-output-file", type=str, default=None,
  #                    help='Output file got from --sample-input-file')
  # group.add_argument("--num-samples", type=int, default=0,
  #                    help='Number of samples to generate unconditionally, '
  #                    'defaults to 0 and interactive conditional sampling')
  # group.add_argument("--genfile", type=str,
  #                    help='Output file when generating unconditionally')
  # group.add_argument("--recompute", action='store_true',
  #                    help='During generation recompute all attention '
  #                    'instead of using previously computed keys/values.')

  return parser


origin_init_process_group = dist.init_process_group


def init_process_group_365(
    backend,
    init_method=None,
    timeout=timedelta(minutes=30),
    world_size: int = -1,
    rank: int = -1,
    store=None,
    group_name: str = "",
    # pg_options= None,
):
  # 设置比较长的nccl等待时间
  timeout = timedelta(minutes=1)
  #    import inspect
  #    print_rank_0(inspect.signature(origin_init_process_group))
  #    print_rank_0(inspect.getabsfile(origin_init_process_group))
  #    print(origin_init_process_group)
  return origin_init_process_group(backend,
                                   init_method=init_method,
                                   timeout=timeout,
                                   world_size=world_size,
                                   rank=rank,
                                   store=store,
                                   group_name=group_name)


def main():
  """Main program."""
  dist.init_process_group = init_process_group_365

  initialize_megatron(extra_args_provider=add_text_generate_args)

  args = get_args()
  if args.num_layers_per_virtual_pipeline_stage is not None:
    print(
        "Interleaved pipeline schedule is not yet supported for text generation."
    )
    exit()

  # Set up model and load checkpoint.
  model = get_model(model_provider)

  if args.load is not None:
    _ = load_checkpoint(model, None, None)

  assert len(model) == 1, "Above condition should have caught this"
  model = model[0]

  prompts = [
      '中国的首都是哪座城市?' * 3,
  ]

  # print_rank_0(prompts)
  # print(sfg)
  # dist.barrier()
  for prompt in prompts:
    output = generate_samples_ds(model, prompt, 20)
    print_rank_0(f'prompt: {prompt}, output:{output}')
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

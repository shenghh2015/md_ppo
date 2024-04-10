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
from tools.generate_samples_bloom_base import add_text_generate_args


def model_provider(args, pre_process=True, post_process=True):
  """Build the model."""

  print_rank_0('building GPT model ...')
  # GPTModel.set_input_tensor
  model = GPTModel(num_tokentypes=0,
                   parallel_output=False,
                   pre_process=pre_process,
                   post_process=post_process)

  return model


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
  tokenizer = get_tokenizer()
  generate_samples_interactive(model,
                               args.max_new_tokens,
                               tokenizer,
                               tokenizer.eod,
                               args.seq_length,
                               recompute=args.recompute,
                               greedy=args.greedy,
                               temperature=args.temperature,
                               top_k=args.top_k,
                               top_p=args.top_p,
                               multiturn=False)

  # prompts = [
  #     '中国的首都是哪座城市?', '海南省的省会是哪座城市?', '请讲一个笑话吧。'
  #     '请写一段python代码实现冒泡排序:\n',
  #     '快速借款逾期有多严重',
  #     '征信没有污点，但报告记录显示有过招商查询记录(四次)，对贷款有影响吗',
  #     '信用卡逾期后能房贷吗',
  #     '购买风险型理财产品有何技巧',
  #     '2亿港元多少人民币',
  #     '有信用卡能在哪个网贷平台贷款',
  #     '银行卡密码忘记了，要重新改密码银行不予办理怎么办?'
  # ]
  # tokenizer = get_tokenizer()
  # outputs = generate_samples(
  #   model,
  #   prompts,
  #   args.max_new_tokens,
  #   tokenizer,
  #   tokenizer.eod,
  #   args.seq_length,
  #   recompute = args.recompute,
  #   temperature=args.temperature,
  #   top_p=args.top_p,
  #   top_k=args.top_k,
  #   greedy=args.greedy,
  #   batch_size=args.micro_batch_size
  #   )

  # for i in range(len(prompts)):
  #   print_rank_0(f'prompt: {prompts[i]}, output:{outputs[i]}')
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

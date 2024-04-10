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
import time
from tools.decorators import log_exe_time

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

sys.path.append('/root/.cache/torch_extensions')

import torch
from megatron import get_args
from megatron import print_rank_0
from megatron import get_tokenizer
from megatron import mpu, prompt_template
from megatron.checkpointing import load_checkpoint
from megatron.initialize import initialize_megatron
from megatron.model.gpt_model import GPTModel
from megatron.training import get_model
from megatron.text_generation_utils import generate_and_write_samples_unconditional
from megatron.text_generation_utils import generate_samples_input_from_file
from megatron.text_generation_utils import generate_samples_interactive
from megatron.text_generation_utils import generate_samples, generate_samples_with_pipeline
from inference.eval_prompts import ten_prompts, financial_prompts, prompts_with_search
# from financial_prompt_extension import get_financial_prompt
from megatron.enums import AttnMaskType
from megatron.model import GPTModel, GPTModelPipe
import deepspeed
# from tools.generate_samples_bloom_base import add_text_generate_args
from tools.extral_args import add_step1_text_generate_args


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

@log_exe_time
def main():
  """Main program."""
  initialize_megatron(extra_args_provider=add_step1_text_generate_args,
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
  #   "请帮马明哲董事长写一篇中国平安保险集团股份有限公司成立35周年的致辞",
  #   "中美博弈的关键在哪里? "
  # ]*4

  # prompts = ten_prompts[:2]
  prompts = ten_prompts
  # prompts = financial_prompts
  tokenizer = get_tokenizer()
  t0 = time.time()
  # 使用pipeline格式
  # outputs = generate_samples_with_pipeline(
  #   model,
  #   prompt_template(prompts),
  #   args.max_new_tokens,
  #   tokenizer,
  #   tokenizer.eod,
  #   args.seq_length,
  #   micro_batch_size=args.micro_batch_size,
  #   pipeline_batch_size=1,
  #   temperature=args.temperature,
  #   top_p=args.top_p,
  #   top_k=args.top_k,
  #   greedy=args.greedy,
  # )

  # 使用正常格式
  outputs = generate_samples(
    model,
    prompt_template(prompts),
    # [p for p in prompts],
    # [p for p in prompts],
    args.max_new_tokens,
    tokenizer,
    tokenizer.eod,
    args.seq_length,
    recompute = args.recompute,
    temperature=args.temperature,
    top_p=args.top_p,
    top_k=args.top_k,
    greedy=args.greedy,
    batch_size=args.micro_batch_size
    )
  print_rank_0('model time:', time.time()-t0)
  if torch.distributed.get_rank() ==0:
    for i in range(len(prompts)):
      print_rank_0(f'prompt: {prompts[i]}, output:{outputs[i]}')
    with open("/tmp/generation_results.txt", "w") as fou:
      fou.writelines(o + "\n" for o in outputs)


# Generate samples.
# if args.num_samples == 0:
#     args.micro_batch_size = 1
#     if args.sample_input_file != None:
#         generate_samples_input_from_file(model)
#     else:
#         generate_samples_interactive(model)
# else:
#     generate_and_write_samples_unconditional(model)


def main_cli_multiturn():
  pass



@log_exe_time
def main_file():
  import json
  file_path = "xxx"
  # model_name = 'bloom-176b-lora-rank-2-3M-data'
  model_name = 'dpo_7b1_test_2000_step'
  output_path = f"eval_{model_name}.test.900.pydict"
  save_batch_size = 64
  # model_name = 'bloom-176b-lora-rank-2-3M-data'
  # model_name = 'bloom-176b-lora-rank-2-3M-data'
  with open(file_path) as f:
    prompt_dicts = [eval(line) for line in f.readlines()]

# def main_file():
  initialize_megatron(extra_args_provider=add_step1_text_generate_args,
                      args_defaults={
                          'tokenizer_type': 'GPT2BPETokenizer',
                          'no_load_rng': True,
                          'no_load_optim': True
                      })

  args = get_args()

  file_path = "sft/llm_wrapper/prompt.pydict"
  # 请使用model uid
  model_name = 'model_011'
  output_path = f"sft/llm_wrapper/eval_{model_name}.wang.900_prompts.4_best_answers_test.pydict"
  save_batch_size = 20
  # sample_num = args.n_best_samples

  with open(file_path) as f:
    try:
      prompt_dicts = [eval(line) for line in f.readlines()]
    except:
      prompt_dicts = [json.loads(line) for line in f.readlines()]

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
  print_rank_0(f'current prompt num: {len(prompt_dicts)}')
  prompt_num = len(prompt_dicts)

  if os.path.exists(output_path):
    if torch.distributed.get_rank() == 0:
      print(f'output file {output_path} exists')
      os.unlink(output_path)

  for batch_start_id in range(0, prompt_num, save_batch_size):
    print_rank_0(f'batch_start_id: {batch_start_id}')
    prompt_dicts_batch = prompt_dicts[batch_start_id:batch_start_id +
                                      save_batch_size]
    prompts = [prompt_dict['prompt'] for prompt_dict in prompt_dicts_batch]
    outputs_list = []
    for i in range(args.n_best_samples):
      # 正常格式
      # outputs = generate_samples(
      #   model,
      #   prompt_template(prompts),
      #   # [p for p in prompts],
      #   # [p for p in prompts],
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

      # pipeline格式
      outputs = generate_samples_with_pipeline(
          model,
          prompt_template(prompts),
          args.max_new_tokens,
          tokenizer,
          tokenizer.eod,
          args.seq_length,
          micro_batch_size=args.micro_batch_size,
          # pipeline_batch_size=len(prompts),
          temperature=args.temperature,
          top_p=args.top_p,
          top_k=args.top_k,
          greedy=args.greedy,
      )
    
    if torch.distributed.get_rank() ==0:
      for i in range(len(prompts)):
        print_rank_0(f'prompt {i}: {prompts[i]}, output:{outputs[i]}')
      
      # print(sfg)
      # 写入到本地
      # assert len(outputs) == len(prompt_dicts_batch),(len(outputs),len(prompt_dicts))
      for prompt_dict, prompt_for_pred, output in zip(
          prompt_dicts_batch, prompts, list(zip(*outputs_list))):
        # prompt_dict['prompt_for_pred'] = prompt_for_pred
        prompt_dict[f'predict_{model_name}'] = output
        with open(output_path, 'a') as f:
          f.write(json.dumps(prompt_dict, ensure_ascii=False))
          f.write('\n')
          f.flush()
        


if __name__ == "__main__":
  main()
  # main_file()

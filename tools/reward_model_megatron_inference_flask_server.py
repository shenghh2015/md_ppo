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
"""基于megatron+flask的统一api接口,默认支持多轮"""

import os
import sys
import json
import time

from megatron import get_args
from megatron import print_rank_0
from megatron import get_tokenizer
from megatron import prompt_template
from megatron import mpu
from megatron.checkpointing import load_checkpoint
from megatron.initialize import initialize_megatron
from megatron.model.gpt_model import GPTModel
from megatron.model.gpt_reward_model import GPTModelCritic
from megatron.training import get_model
from megatron.text_generation_utils import generate_and_write_samples_unconditional
from megatron.text_generation_utils import generate_samples_input_from_file
from megatron.text_generation_utils import generate_samples_interactive
from megatron.text_generation_utils import generate_samples

from megatron.enums import AttnMaskType
from megatron.model import GPTModel, GPTModelPipe
import deepspeed
import threading
import torch.distributed as dist
from datetime import timedelta
from tools.generate_samples_bloom_base import add_text_generate_args
from tools.extral_args import add_step2_train_reward_model_args
from reward_model_megatron_inference import generate_reward_model_values

lock = threading.Lock()
MODEL = None
ARGS = None


def model_provider(args, pre_process=True, post_process=True):
  """Build the model."""

  print_rank_0('building GPT model ...')
  # GPTModel.set_input_tensor
  model = GPTModelCritic(num_tokentypes=0,
                         parallel_output=False,
                         pre_process=pre_process,
                         post_process=post_process)

  return model


def init_flask(port=8050):
  from flask import Flask
  from flask import request
  from flask import jsonify
  app = Flask(__name__)

  @app.route("/reward_model_inference", methods=['POST'])
  def generate():
    post_data = request.get_json()
    print(f'收到请求准备brodcast', {threading.current_thread().name}, post_data)
    t0 = time.time()
    # 单线程执行
    with lock:
      # dist.monitored_barrier(wait_all_ranks=True)
      t1 = time.time()
      dist.broadcast_object_list([post_data], 0)
      output = query(post_data)
      t2 = time.time()
      output['model_exe_time'] = t2 - t0
      output['wait_to_exe_time'] = t1 - t0
    ret = jsonify(**output)
    # finish_evet.clear()
    # output = query(post_data)
    return ret

  # app.run('0.0.0.0',port=8501,debug=False,threaded=True)
  app.run('0.0.0.0', port=port, debug=False, threaded=True)


post_data_template = {
    'user_input': '海口是',
    'temperature': 0.8,
    'top_p': 0.95,
    'max_gen_len': 256
}


def query(post_data, replace_line_break=False):
  args = get_args()
  # ser_input,temperature,top_p,max_gen_len=256
  user_input = post_data['user_input']
  use_prompt_template = post_data.get('use_prompt_template', True)
  # input_split = user_input.split()
  # assert len(input_split) == 2
  assert isinstance(user_input, list) and len(user_input) > 0
  if use_prompt_template:
    user_input = [
        prompt_template(i['prompt'], i['answer']) for i in user_input
    ]
  # user_input = json.loads(user_input) # [(q1,a1),(q2,a2)], list(tuple)
  # last_user_input = user_input[-1]
  # assert len(user_input[-1]) == 1
  # user_input[-1] = (user_input[-1][0],"")
  # input = [prompt_template(i[0],ans=i[1]) for i in user_input]
  # user_input = '\n'.join(input)
  print(f'user input: {user_input[0]}, length of user input:{len(user_input)}')
  # last_input = f"Human:{user_input[-1][0]} \nAssitant:"
  # input_str = "".join(input) + last_input
  # temperature = post_data.get('temperature', post_data_template['temperature'])
  # top_p = post_data.get('top_p', post_data_template['temperature'])
  # max_gen_len = post_data.get('max_gen_len', post_data_template['max_gen_len'])
  # # if not isinstance(user_input,list):
  # #     user_input = [user_input]

  # # ARGS.seq_len =
  # generate_txt = generate_samples(MODEL, user_input)[0]
  # temperature = post_data.get('temperature', args.temperature)
  # top_p = post_data.get('top_p', args.top_p)
  # top_k = post_data.get('top_k',args.top_k)
  # max_new_tokens = post_data.get('max_new_tokens', args.max_new_tokens)
  # greedy  =post_data.get('greedy',args.greedy)
  # prmopt = post_data.get('prompt', )
  # answer = post_data.get('answer', )
  # if not isinstance(user_input,list):
  #     user_input = [user_input]

  # ARGS.seq_len =
  tokenizer = get_tokenizer()
  # generate_txt = generate_samples(
  #   MODEL,
  #   user_input,
  #   max_new_tokens,
  #   tokenizer,
  #   tokenizer.eod,
  #   args.seq_length,
  #   recompute = args.recompute,
  #   temperature=temperature,
  #   top_p=top_p,
  #   top_k=top_k,
  #   greedy=greedy
  #   )
  outputs = generate_reward_model_values(MODEL,
                                         user_input,
                                         tokenizer,
                                         tokenizer.eod,
                                         args.seq_length,
                                         batch_size=args.micro_batch_size)
  # test_intput = ["Human:中国的首都是哪里？\nAssistant:北京。"]
  # test_output = generate_reward_model_values(
  #   MODEL,
  #   test_intput,
  #   tokenizer,
  #   tokenizer.eod,
  #   args.seq_length,
  #   batch_size=args.micro_batch_size
  #   )
  # print(f'{test_output}')
  # if outputs is None:
  #   outputs = generate_reward_model_values(MODEL, f"Human:{last_user_input[0]}\nAssitant:")[0]
  if outputs is None:
    outputs = "请刷新页面或重新提问"
  # if generate_txt[0] == "\"" and generate_txt[-1] == "\"":
  #   generate_txt.lstrip("\"").strip("\"")
  # generate_txt = generate_txt.lstrip(" ？\n").strip("</s>")
  #, max_gen_len=max_gen_len, temperature=temperature, top_p=top_p)[0]
  # if replace_line_break:
  #   generate_txt = generate_txt.replace('\n', '<br>')
  output = {"rewards": outputs}
  return output


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
  timeout = timedelta(days=365)
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

  initialize_megatron(extra_args_provider=add_step2_train_reward_model_args)

  args = get_args()
  global ARGS
  ARGS = args
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
  global MODEL
  MODEL = model
  rank = dist.get_rank()
  if dist.get_rank() == 0:
    print('init flask server ')
    init_flask(port=ARGS.port)

  while True:
    if rank > 0:
      # print('等待broadcast')
      # dist.barrier()
      # 持续等待
      # dist.monitored_barrier(timeout=3600*1e8)
      post_data = [None]
      dist.broadcast_object_list(post_data, 0)
      # print('收到数据计算中')
      # dist.barrier()
      post_data_template.update(**post_data[0])
      query(post_data_template)


if __name__ == "__main__":
  main()

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
"""基于megatron+flask的统一api接口,默认支持多轮 + 流式输出"""

import os
import sys
import json
import time
import queue
import threading
sys.setrecursionlimit(10**5)
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

sys.path.append('/root/.cache/torch_extensions')

from megatron import get_args
from megatron import print_rank_0
from megatron import get_tokenizer
from megatron import prompt_template
from megatron import mpu
from megatron.checkpointing import load_checkpoint
from megatron.initialize import initialize_megatron
from megatron.model.gpt_model import GPTModel
from megatron.training import get_model
from megatron.text_generation_utils import generate_and_write_samples_unconditional
from megatron.text_generation_utils import generate_samples_input_from_file
from megatron.text_generation_utils import generate_samples_interactive
from megatron.text_generation_utils import generate_samples, generate_samples_stream
# from flask_socketio import SocketIO
# from flask_socketio import send, emit
from gevent import monkey

monkey.patch_all()
from flask import Flask, Response, render_template, stream_with_context
from gevent.pywsgi import WSGIServer

from megatron.enums import AttnMaskType
from megatron.model import GPTModel, GPTModelPipe
import deepspeed
import threading
import torch.distributed as dist
from datetime import timedelta
from tools.generate_samples_bloom_base import add_text_generate_args

lock = threading.Lock()
lock_single_requests = threading.Lock()
lock_req_q_put_or_get = threading.Lock()
MODEL = None
ARGS = None
ret_q = queue.Queue()
client_queue = queue.Queue()


class StartFlag:
  pass


class EndFlag:
  pass


class StreamTokenizer:
  """
    from https://github.dev/hyperonym/basaran
    StreamTokenizer wraps around a tokenizer to support stream decoding.
    """
  def __init__(self, tokenizer):
    super().__init__()
    self.tokenizer = tokenizer
    self.replacement = chr(0xFFFD)
    self.buffer = []
    self.surrogates = 0
    self.start = 0
    self.end = 0

  def decode(self, token):
    """Decode token to string while handling surrogates and whitespace."""

    # <unk>, <pad> and other special tokens will be decoded into ''.
    text = self.tokenizer.decode(token, skip_special_tokens=True)

    # Handle replacement characters caused by multi-byte-pair-encoding or
    # Unicode surrogates or multi-code-point graphemes like emojis.
    if self.replacement in text:
      n = -self.surrogates if self.surrogates > 0 else len(self.buffer)
      tokens = self.buffer[n:] + [token]
      text = self.tokenizer.decode(tokens, skip_special_tokens=True)

      # Check whether the last grapheme was successfully decoded.
      if text and text[-1] != self.replacement:
        text = text.replace(self.replacement, "")
        self.surrogates = 0
      else:
        text = ""
        self.surrogates += 1
    else:
      self.surrogates = 0

    # Handle whitespace between tokens.
    tokens = self.buffer + [token]
    prefix = self.tokenizer.decode(self.buffer, skip_special_tokens=True)
    whole = self.tokenizer.decode(tokens, skip_special_tokens=True)
    if prefix + " " + text == whole:
      text = " " + text

    # Update buffer and offsets.
    self.buffer = self.buffer[-4:] + [token]
    self.start = self.end
    self.end += len(text)

    return text


def rank_0_query_helper(post_data, ret_q: queue.Queue,
                        client_queue: queue.Queue):
  with lock:
    # 可能存在历史信息, 需要去掉
    # while not ret_q.empty():
    #   ret_q.get()
    stream_tokenzer = StreamTokenizer(get_tokenizer().tokenizer)

    output_stream = query(post_data)
    ret_q.put(StartFlag)
    is_client_close = False
    for output in output_stream:
      output = stream_tokenzer.decode(output[0])
      # print('put',output)
      ret_q.put(output)
      # 如果客户端长时间没有回应，不管客户端
      if not is_client_close:
        try:
          client_queue.get(timeout=5)
        except:
          is_client_close = True
      # time.sleep(0.1)

    # 最后给一个结束符
    ret_q.put(EndFlag)
    del stream_tokenzer


def model_provider(args, pre_process=True, post_process=True):
  """Build the model."""

  print_rank_0('building GPT model ...')
  # GPTModel.set_input_tensor
  model = GPTModel(num_tokentypes=0,
                   parallel_output=False,
                   pre_process=pre_process,
                   post_process=post_process)
  return model


def init_flask(port=8050):
  from flask import Flask
  from flask import request
  from flask import jsonify
  app = Flask(__name__)
  # app.config['SECRET_KEY'] = 'secret!'
  # socketio = SocketIO(app)

  # @app.route("/generate", methods=['POST'])
  # def generate():
  #   post_data = request.get_json()
  #   print(f'收到请求准备brodcast', {threading.current_thread().name}, post_data)
  #   # 单线程执行
  #   with lock:
  #     # dist.monitored_barrier(wait_all_ranks=True)
  #     dist.broadcast_object_list([post_data], 0)
  #     output = query(post_data)
  #   ret = jsonify(**output)
  #   # finish_evet.clear()
  #   # output = query(post_data)
  #   return ret

  # @socketio.on('generate')
  # def handle_generate_event(json):
  #     print('received json: ' + str(json))
  #     send(json, json=True)
  #     with lock:
  #       # dist.monitored_barrier(wait_all_ranks=True)
  #       dist.broadcast_object_list([post_data], 0)
  #       output = query(post_data)
  # app.run('0.0.0.0',port=8501,debug=False,threaded=True)

  @app.route("/generate", methods=['POST'])
  def generate():
    post_data = request.get_json()
    print(f'收到请求准备brodcast', {threading.current_thread().name}, post_data)

    def respond_to_client():
      with lock_single_requests:
        dist.broadcast_object_list([post_data], 0)
        # stream_tokenzer = StreamTokenizer(get_tokenizer().tokenizer)
        # output_stream = query(post_data)

        t = threading.Thread(target=rank_0_query_helper,
                             args=(post_data, ret_q, client_queue))
        t.start()
        start_flag = False
        while True:
          output = ret_q.get()
          # print('get',output)
          if not start_flag:
            if output is StartFlag:
              start_flag = True
            continue

          client_queue.put(None)
          if output is EndFlag:
            break

          _data = json.dumps({"generated_text": output})
          # print('send to client',output)
          yield f"id: 1\ndata: {_data}\nevent: online\n\n"

    return Response(respond_to_client(), mimetype='text/event-stream')

  args = get_args()
  http_server = WSGIServer(("0.0.0.0", port), app)
  http_server.serve_forever()
  

  # socketio.run(app, host='0.0.0.0', debug=False,port=args.port)
  # app.run('0.0.0.0', port=port, debug=False, threaded=True)


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
  # user_input = json.loads(user_input) # [(q1,a1),(q2,a2)], list(tuple)
  last_user_input = user_input[-1]
  assert len(user_input[-1]) == 1
  user_input[-1] = (user_input[-1][0], "")
  input = [prompt_template(i[0], ans=i[1]) for i in user_input]
  user_input = '</s>'.join(input)
  print_rank_0(f'user input: {user_input}')
  # last_input = f"Human:{user_input[-1][0]} \nAssitant:"
  # input_str = "".join(input) + last_input
  # temperature = post_data.get('temperature', post_data_template['temperature'])
  # top_p = post_data.get('top_p', post_data_template['temperature'])
  # max_gen_len = post_data.get('max_gen_len', post_data_template['max_gen_len'])
  # # if not isinstance(user_input,list):
  # #     user_input = [user_input]

  # # ARGS.seq_len =
  # generate_txt = generate_samples(MODEL, user_input)[0]
  temperature = post_data.get('temperature', args.temperature)
  top_p = post_data.get('top_p', args.top_p)
  top_k = post_data.get('top_k', args.top_k)
  max_new_tokens = post_data.get('max_new_tokens', args.max_new_tokens)
  greedy = post_data.get('greedy', args.greedy)
  # if not isinstance(user_input,list):
  #     user_input = [user_input]

  # ARGS.seq_len =
  tokenizer = get_tokenizer()
  yield from generate_samples_stream(MODEL,
                                     user_input,
                                     max_new_tokens,
                                     tokenizer,
                                     tokenizer.eod,
                                     args.seq_length,
                                     recompute=False,
                                     temperature=temperature,
                                     top_p=top_p,
                                     top_k=top_k,
                                     greedy=greedy)
  # if generate_txt is None:
  #   generate_txt = generate_samples(MODEL, f"Human:{last_user_input[0]}\nAssitant:")[0]
  #   if generate_txt is None:
  #     generate_txt = "当前输入超出最大字数限制，请刷新页面或重新提问"
  # if generate_txt[0] == "\"" and generate_txt[-1] == "\"":
  #   generate_txt.lstrip("\"").strip("\"")
  # generate_txt = generate_txt.lstrip(" ？\n").strip("</s>")
  # #, max_gen_len=max_gen_len, temperature=temperature, top_p=top_p)[0]
  # if replace_line_break:
  #   generate_txt = generate_txt.replace('\n', '<br>')
  # output = {"generated_text": generate_txt}
  # return output


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

  initialize_megatron(extra_args_provider=add_text_generate_args)

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
      for _ in query(post_data_template):
        continue


if __name__ == "__main__":
  main()

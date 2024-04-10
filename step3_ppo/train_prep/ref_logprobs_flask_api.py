"""
实现ppo算法中的引用模型接口, 并以hf的模型为基础实现。
这里模型的输入为主要返回lm logits
"""

import threading
from transformers import AutoTokenizer, AutoModelForCausalLM
from datetime import timedelta
import torch
from flask import Flask
from flask import request
from flask import jsonify
import argparse
from torch.nn.utils.rnn import pad_sequence
import time
from step3_ppo.models.gpt_model_ppo import GPTModelWithPPOValueHead
logprobs_of_labels = GPTModelWithPPOValueHead.logprobs_of_labels
lock = threading.Lock()


TOKENIZER = None
MODEL = None
ARGS = None



def parse_args():
  parser = argparse.ArgumentParser(description='cmd for api')
  parser.add_argument('--batch_size', type=int, default=32)
  parser.add_argument('--model_name', type=str, required=True)
  parser.add_argument('--port', type=str, default='8050')

  return parser.parse_args()

def init_model():
  # global TOKENIZER
  global MODEL
  if MODEL is not None:
    return TOKENIZER,MODEL
  model_name = ARGS.model_name
  # tokenizer = AutoTokenizer.from_pretrained(model_name)
  model = AutoModelForCausalLM.from_pretrained(model_name,torch_dtype=torch.float16).cuda()
  # tokenizer.padding_side = 'right'
  # TOKENIZER = tokenizer
  MODEL = model
  # MODEL.eval()
  return TOKENIZER,MODEL

def init_flask(port=8050):
  
  app = Flask(__name__)

  @app.route("/ref_model_logits", methods=['POST'])
  def ref_model_logits():
    t0 = time.time()
    post_data = request.get_json()
    print(f'收到请求', {threading.current_thread().name}, post_data)
    # 单线程执行
    with lock:
      # dist.monitored_barrier(wait_all_ranks=True)
      output = query(post_data)
    output['model_exe_time'] = time.time()-t0
    ret = jsonify(**output)
    # finish_evet.clear()
    # output = query(post_data)
    
    return ret

  # app.run('0.0.0.0',port=8501,debug=False,threaded=True)
  app.run('0.0.0.0', port=port, debug=False, threaded=True)


def query(post_data):
  """

  Args:
      post_data (_type_): _description_
      replace_line_break (bool, optional): _description_. Defaults to False.

  Returns:
      _type_: _description_
  """
  batch_size = ARGS.batch_size
  assert isinstance(post_data['query_tensors'],list),post_data['query_tensors']
  # assert isinstance(post_data['query_tensors'][0],int), post_data['query_tensors'][0]

  query_lens = [len(query_tensor) for query_tensor in post_data['query_tensors']]
  respond_lens = [len(response_tensor) for response_tensor in post_data['response_tensors']]
  input_ids_list = [
    torch.LongTensor(query_tensor+response_tensor).cuda()
    for query_tensor,response_tensor in zip(post_data['query_tensors'],post_data['response_tensors'])
    ]
  # ser_input,temperature,top_p,max_gen_len=25
  input_ids = pad_sequence(input_ids_list,batch_first=True,padding_value=0).cuda()
  
  logits_list = []
  # assert isinstance(user_input,list) and isinstance(user_input[0],str),user_input
  for batch_id in range(0,len(post_data['query_tensors']),batch_size):
    query_lens_batch = query_lens[batch_id:batch_id+batch_size]
    respond_lens_batch = respond_lens[batch_id:batch_id+batch_size]
    input_ids_batch = input_ids[batch_id:batch_id+batch_size]
    with torch.no_grad():
      output = MODEL(input_ids=input_ids_batch,return_dict=True)
    logits = output['logits']
    for i in range(logits.shape[0]):
      q_len = query_lens_batch[i]
      res_len = respond_lens_batch[i]
      start = q_len - 1
      end = res_len + start
      # end = start + respond_lens_batch[i]
      logits_list.append([
        logprobs_of_labels(
          logits[i,start:end,:],
          input_ids_batch[i,q_len:q_len+res_len]
        ).tolist()
      ]
        )
  
  return {'lm_logtis':logits_list}


def main():
  global ARGS
  ARGS = parse_args()
  print('init model')
  init_model()
  print('init flask')
  init_flask(port=ARGS.port)


if __name__ == "__main__":
  main()






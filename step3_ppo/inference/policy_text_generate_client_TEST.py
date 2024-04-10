import pkg_resources
import sys, os
from pkg_resources import DistributionNotFound, VersionConflict
import multiprocessing
from palframe.nlp import pydict_file_read, pydict_file_write
import argparse
dependencies = [
    'sseclient-py==1.7.2',
]
try:
  pkg_resources.require(dependencies)
except:
  print('sseclient-py missing, installing')
  os.system(f'{sys.executable} -m pip install sseclient-py==1.7.2')

import sseclient
import json
import requests
from datetime import date
from tqdm import tqdm

import balancer_config as config

def generate(endpiont, data):
  r = requests.post(endpiont,
                    json=data,
                    stream=True,
                    headers={'Accept': 'text/event-stream'})

  client = sseclient.SSEClient(r)

  output_seq = ""
  for event in client.events():
    output = json.loads(event.data)['generated_text']
    history_output_ = output
    output = history_output_
    output_seq += output

  output_seq += '\n'
  return output_seq

def process_data(url, datalist):
  _datalist = []
  for i, data in enumerate(tqdm(datalist)):
    prompt = data['conversations'][0]['value']
    input_data = {'user_input': [(prompt,)], "temperature": 0.7, "greedy": False}
    generated_text = generate(url, input_data)
    _data = {
      'conversations': data['conversations'],
      'lm_response': generated_text
    }
    _datalist.append(_data)
    if i % 10 == 0: 
      print('*'*50)
      print(f'prompt: {prompt} \n response: {generated_text}')
  return _datalist

def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--data_path", type=str, default=None)
  parser.add_argument("--lm_infer_url", type=str, default=None)
  parser.add_argument("--output_root", type=str, default=None)
  parser.add_argument("--output_tag", type=str, default='ppo_infer')
  parser.add_argument("--data_num", type=int, default=None)
  return parser.parse_args()

def main():
  
  args = get_args()

  # read data from pydict datapath
  data_iter = pydict_file_read(args.data_path)
  datalist = []
  for i, data in enumerate(data_iter):
    if i == args.data_num: break
    datalist.append(data)
  
  # inference process
  output_list = process_data(args.lm_infer_url, datalist)

  # save results
  input_name = os.path.basename(args.data_path).split('.pydict')[0]
  output_filename = f'{input_name}-{args.output_tag}-{args.data_num}.pydict'
  output_path = os.path.join(args.output_root, output_filename)
  pydict_file_write(output_list, output_path)

if __name__ == "__main__":
    main()
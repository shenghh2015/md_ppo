from palframe.nlp import pydict_file_read, pydict_file_write
import os
from tqdm import tqdm
import argparse
from megatron.tokenizer import build_tokenizer
from megatron import prompt_template
import numpy as np


def analyze_len_stats(len_list):
  mean, std = np.mean(len_list), np.std(len_list)
  max_len, min_len = max(len_list), min(len_list)
  print(">> token sequence stats:")
  print(f"total seqs: {len(len_list)}")
  print(
      f"lengths: mean {round(mean,2)}, std {round(std,2)}, min {min_len}, max {max_len}"
  )


def get_prompt_token_len(tokenizer, prompt):
  prompt = prompt_template(prompt)
  prompt_ids = tokenizer.tokenize(prompt)
  return len(prompt_ids)


def extract_prompts(datalist, split="train"):
  _datalist = []
  _cache = {}
  for i, data in enumerate(tqdm(datalist)):
    if data['split'] == split:
      _data = {'conversations': data['conversations']}
      prompt = _data['conversations'][0]['value']
      if not prompt in _cache:
        _cache[prompt] = _data
      if i % 10000 == 0: print(f'>> {i} processed!')
  _datalist = list(_cache.values())
  return _datalist


def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--data_path", type=str, default=None)
  parser.add_argument("--output_dir", type=str, default=None)
  # tokenizer
  group = parser.add_argument_group(title='tokenizer')
  group.add_argument('--tokenizer-type',
                     type=str,
                     required=True,
                     choices=[
                         'BertWordPieceLowerCase', 'BertWordPieceCase',
                         'GPT2BPETokenizer', 'PretrainedFromHF'
                     ],
                     help='What type of tokenizer to use.')
  group.add_argument('--vocab-file',
                     type=str,
                     default=None,
                     help='Path to the vocab file')
  group.add_argument('--merge-file',
                     type=str,
                     default=None,
                     help='Path to the BPE merge file (if necessary).')
  group.add_argument('--append-eod',
                     action='store_true',
                     help='Append an <eod> token to the end of a document.')
  group.add_argument('--append-ans',
                     action='store_true',
                     help='Append an <ans> token to the end of a prefix.')
  group.add_argument('--use-prompt-template',
                     action='store_true',
                     help='Use prompt template in prompt')
  group.add_argument("--tokenizer-name-or-path",
                     type=str,
                     default=None,
                     help="Name or path of the huggingface tokenizer.")
  group.add_argument('--make-vocab-size-divisible-by',
                     type=int,
                     default=128,
                     help='Pad the vocab size to be divisible by this value.'
                     'This is added for computational efficieny reasons.')
  group.add_argument(
      '--pad-vocab-size-to',
      type=int,
      default=None,
      help='Pad the vocab size to be divisible by this value.'
      'Value of the size of the vocabulary of the tokenizer to reach. This value must be greater than'
      ' the initial size of the tokenizer. If this argument is used the value of '
      '`make-vocab-size-divisible-by` will be ignored.')

  parser.add_argument("--rank", type=int, default=0)
  parser.add_argument("--val_num", type=int, default=40)
  parser.add_argument("--max_len", type=int, default=500)
  args = parser.parse_args()
  args.rank = 0
  args.tensor_model_parallel_size = 1
  args.vocab_extra_ids = 0
  return args


def main():

  args = get_args()
  data_path = args.data_path
  output_dir = args.output_dir
  print(output_dir)
  os.makedirs(output_dir, exist_ok=True)

  # tokenizer
  tokenizer = build_tokenizer(args)

  # extract train prompts
  input_data = pydict_file_read(data_path)
  datalist = extract_prompts(input_data, split="train")
  len_list = []
  _datalist = []
  for data in datalist:
    token_len = get_prompt_token_len(tokenizer,
                                     data["conversations"][0]["value"])
    if token_len < args.max_len:
      _datalist.append(data)
      len_list.append(token_len)
  train_filename = os.path.basename(
      data_path) + f"_ppo_prompts_train.{len(datalist)}.pydict"
  output_path = os.path.join(output_dir, train_filename)
  pydict_file_write(_datalist, output_path)
  print(output_path)
  analyze_len_stats(len_list)

  # extract eval prompts
  input_data = pydict_file_read(data_path)
  datalist = extract_prompts(datalist=input_data, split="test")
  # datalist = [data for data in datalist if get_prompt_token_len(tokenizer, data["conversations"][0]["value"]) < args.max_len]
  len_list = []
  _datalist = []
  for data in datalist:
    prompt = data["conversations"][0]["value"]
    token_len = get_prompt_token_len(tokenizer=tokenizer, prompt=prompt)
    if token_len < args.max_len:
      _datalist.append(data)
      len_list.append(token_len)
  datalist = _datalist[:args.val_num]
  val_filename = os.path.basename(
      data_path) + f"_ppo_prompts_eval.{len(datalist)}.pydict"
  output_path = os.path.join(output_dir, val_filename)
  pydict_file_write(datalist, output_path)
  print(output_path)
  analyze_len_stats(len_list[:args.val_num])


if __name__ == '__main__':
  main()

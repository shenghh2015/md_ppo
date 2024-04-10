from megatron import prompt_template
from megatron.data.indexed_dataset import best_fitting_sft_dtype
from megatron.tokenizer import build_tokenizer
from megatron.data import indexed_dataset
import argparse
import json
import multiprocessing
import os
import sys
import time
import torch

try:
  import nltk
  nltk_available = True
except ImportError:
  nltk_available = False

# https://stackoverflow.com/questions/33139531/preserve-empty-lines-with-nltks-punkt-tokenizer
class CustomLanguageVars(nltk.tokenize.punkt.PunktLanguageVars):

  _period_context_fmt = r"""
        \S*                          # some word material
        %(SentEndChars)s             # a potential sentence ending
        \s*                       #  <-- THIS is what I changed
        (?=(?P<after_tok>
            %(NonWord)s              # either other punctuation
            |
            (?P<next_tok>\S+)     #  <-- Normally you would have \s+ here
        ))"""


class IdentitySplitter(object):
  def tokenize(self, *text):
    return text


class Encoder(object):
  def __init__(self, args):
    self.args = args

  def initializer(self):
    # Use Encoder class as a container for global data
    Encoder.tokenizer = build_tokenizer(self.args)
    if self.args.split_sentences:
      if not nltk_available:
        print("NLTK is not available to split sentences.")
        exit()
      splitter = nltk.load("tokenizers/punkt/english.pickle")
      if self.args.keep_newlines:
        # this prevents punkt from eating newlines after sentences
        Encoder.splitter = nltk.tokenize.punkt.PunktSentenceTokenizer(
            train_text=splitter._params, lang_vars=CustomLanguageVars())
      else:
        Encoder.splitter = splitter

    else:
      Encoder.splitter = IdentitySplitter()

  def encode_split_prompt(self, json_line):
    # out data process method
    data = eval(json_line)
    input_key = 'chats'
    ids = {}
    doc_ids = []
    # prompt_mask = []

    text_list = data[input_key]
    text_tensor_list = [
        torch.tensor(Encoder.tokenizer.tokenize(text)) for text in text_list
    ]
    text_ids_list = [
        text_tensor[text_tensor != Encoder.tokenizer.eod].tolist()
        for text_tensor in text_tensor_list
    ]
    text_ids_list = [i for i in text_ids_list if i]

    assert all([len(i) > 0 for i in text_ids_list]), (text_list, text_ids_list)

    # 去掉非偶数轮的语料
    if len(text_ids_list) % 2 != 0:
      return {'text': []}, len(json_line)
    for i, text_ids in enumerate(text_ids_list):
      doc_ids.extend(text_ids)
      if (i + 1) % 2 == 1:
        doc_ids.append(-200)  # 用来指示prompt的位置
        # prompt_mask.extend([1]*len(text_ids))
      else:
        # prompt_mask.extend([0]*len(text_ids))
        if i < (len(text_ids_list) - 1):
          # 单轮轮对话中插入一个-100特殊token,最后一轮不用加, 后面有一个额外的eod
          doc_ids.append(-100)
          # prompt_mask.append(0)

    doc_ids.append(Encoder.tokenizer.eod)

    ids['text'] = [doc_ids]

    assert (torch.tensor(ids['text'][-1]) == Encoder.tokenizer.eod).sum() == 1

    return ids, len(json_line)

  def encode_wo_split_prompt(self, json_line):
    """
    不区分prompt和ans,但是要求输入数据的轮次为偶数
    """
    # out data process method
    data = eval(json_line)
    input_key = 'chats'
    ids = {}
    doc_ids = []
    # prompt_mask = []

    text_list = data[input_key]
    text_tensor_list = [
        torch.tensor(Encoder.tokenizer.tokenize(text)) for text in text_list
    ]
    # text_tensor_list = [i for i in text_tensor_list if i] # 去空
    # text_tensor_list = torch.tensor(text_ids)
    text_ids_list = [
        text_tensor[text_tensor != Encoder.tokenizer.eod].tolist()
        for text_tensor in text_tensor_list
    ]
    text_ids_list = [i for i in text_ids_list if i]

    assert all([len(i) > 0 for i in text_ids_list]), (text_list, text_ids_list)

    # 去掉非偶数轮的语料
    if len(text_ids_list) % 2 != 0:
      return {'text': []}, len(json_line)

    for i, text_ids in enumerate(text_ids_list):
      doc_ids.extend(text_ids)
      if (i + 1) % 2 == 0 and i < (len(text_ids_list) - 1):
        # 单轮轮对话中插入一个-100特殊token,最后一轮不用加, 后面有一个额外的eod
        doc_ids.append(-100)

    doc_ids.append(Encoder.tokenizer.eod)

    ids['text'] = [doc_ids]

    assert (torch.tensor(ids['text'][-1]) == Encoder.tokenizer.eod).sum() == 1

    return ids, len(json_line)

  def encode_belle_3m5_split_prompt(self, json_line):
    # out data process method
    data = eval(json_line)
    # input_key = 'conversations'
    ids = {}
    doc_ids = []
    # prompt_mask = []
    conversations = data["conversations"]
    assert len(conversations) % 2 == 0, conversations
    # text_list = [conversation['value'] for conversation in conversations]

    text_list = [[
        prompt_template(conversations[i]['value']),
        conversations[i + 1]['value']
    ] for i in range(0, len(conversations), 2)]
    text_list = sum(text_list, [])

    text_tensor_list = [
        torch.tensor(Encoder.tokenizer.tokenize(text)) for text in text_list
    ]
    # text_tensor_list = [i for i in text_tensor_list if i] # 去空
    # text_tensor_list = torch.tensor(text_ids)
    text_ids_list = [
        text_tensor[text_tensor != Encoder.tokenizer.eod].tolist()
        for text_tensor in text_tensor_list
    ]
    text_ids_list = [i for i in text_ids_list if i]

    # assert all([len(i) > 0 for i in text_ids_list]), (text_list,text_ids_list)
    if len(text_ids_list) != len(text_list):
      return {'text': []}, len(json_line)

    assert len(text_ids_list) == len(text_list), (text_list, text_ids_list)

    # # 去掉非偶数轮的语料
    # if len(text_ids_list) % 2 != 0:
    #   return {'text':[]},len(json_line)
    for i, text_ids in enumerate(text_ids_list):
      doc_ids.extend(text_ids)
      if (i + 1) % 2 == 1:
        doc_ids.append(-200)  # 用来指示prompt的位置
        # prompt_mask.extend([1]*len(text_ids))
      else:
        # prompt_mask.extend([0]*len(text_ids))
        if i < (len(text_ids_list) - 1):
          # 单轮轮对话中插入一个-100特殊token,最后一轮不用加, 后面有一个额外的eod
          doc_ids.append(-100)
          # prompt_mask.append(0)

    doc_ids.append(Encoder.tokenizer.eod)

    ids['text'] = [doc_ids]

    assert (torch.tensor(ids['text'][-1]) == Encoder.tokenizer.eod).sum() == 1

    return ids, len(json_line)

  encode = encode_belle_3m5_split_prompt

  def __encode(self, json_line):
    # out data process method
    data = eval(json_line)
    output_key = 'text'
    input_key = 'chats'
    ids = {}
    doc_ids = []
    prompt_mask = []

    text_list = data[input_key]
    text_tensor_list = [
        torch.tensor(Encoder.tokenizer.tokenize(text)) for text in text_list
    ]
    # text_tensor_list = [i for i in text_tensor_list if i] # 去空
    # text_tensor_list = torch.tensor(text_ids)
    text_ids_list = [
        text_tensor[text_tensor != Encoder.tokenizer.eod].tolist()
        for text_tensor in text_tensor_list
    ]
    text_ids_list = [i for i in text_ids_list if i]

    assert all([len(i) > 0 for i in text_ids_list]), (text_list, text_ids_list)

    # 去掉非偶数轮的语料
    if len(text_ids_list) % 2 != 0:
      return {'text': [], 'prompt_mask': []}, len(json_line)
    for i, text_ids in enumerate(text_ids_list):
      doc_ids.extend(text_ids)
      if i % 2 == 1:
        prompt_mask.extend([1] * len(text_ids))
      else:
        prompt_mask.extend([0] * len(text_ids))
        if i < (len(text_ids_list) - 1):
          # 单轮轮对话中插入一个-100特殊token,最后一轮不用加, 后面有一个额外的eod
          doc_ids.append(-100)
          prompt_mask.append(0)

    doc_ids.append(Encoder.tokenizer.eod)
    prompt_mask.append(0)

    ids['text'] = [doc_ids]
    ids['prompt_mask'] = [prompt_mask]

    assert (torch.tensor(ids['text'][-1]) == Encoder.tokenizer.eod).sum() == 1

    assert len(doc_ids) == len(prompt_mask), (text_list, text_ids_list)

    return ids, len(json_line)


def get_args():
  parser = argparse.ArgumentParser()
  group = parser.add_argument_group(title='input data')
  group.add_argument('--input', type=str, help='Path to input JSON')
  group.add_argument('--datasets',
                     nargs='+',
                     default=None,
                     help='Paths to one or more input datasets to merge')
  group.add_argument('--json-keys',
                     nargs='+',
                     default=['text'],
                     help='space separate listed of keys to extract from json')
  group.add_argument('--split-sentences',
                     action='store_true',
                     help='Split documents into sentences.')
  group.add_argument('--keep-newlines',
                     action='store_true',
                     help='Keep newlines between sentences when splitting.')

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

  group = parser.add_argument_group(title='output data')
  group.add_argument('--output-prefix',
                     type=str,
                     required=True,
                     help='Path to binary output file without suffix')
  group.add_argument('--dataset-impl',
                     type=str,
                     default='mmap',
                     choices=['lazy', 'cached', 'mmap'])

  group = parser.add_argument_group(title='runtime')
  group.add_argument('--workers',
                     type=int,
                     default=1,
                     help='Number of worker processes to launch')
  group.add_argument('--log-interval',
                     type=int,
                     default=100,
                     help='Interval between progress updates')
  args = parser.parse_args()
  args.keep_empty = False

  if args.tokenizer_type.lower().startswith('bert'):
    if not args.split_sentences:
      print(
          "Bert tokenizer detected, are you sure you don't want to split sentences?"
      )

  # some default/dummy values for the tokenizer
  args.rank = 0
  args.tensor_model_parallel_size = 1
  args.vocab_extra_ids = 0

  return args


def main():
  args = get_args()
  startup_start = time.time()

  print("Opening", args.input)
  # output_key = 'text'
  fin = open(args.input, 'r', encoding='utf-8')

  if nltk_available and args.split_sentences:
    nltk.download("punkt", quiet=True)

  encoder = Encoder(args)
  tokenizer = build_tokenizer(args)
  pool = multiprocessing.Pool(args.workers, initializer=encoder.initializer)
  encoded_docs = pool.imap(encoder.encode, fin, 25)
  # print('encoded_docs num',len(encoded_docs))
  # encoded_docs = [encoded_doc for encoded_doc in encoded_docs if encoded_doc[0] is not None]
  # print('encoded_docs num after filter',len(encoded_docs))
  #encoded_docs = map(encoder.encode, fin)

  level = "document"
  if args.split_sentences:
    level = "sentence"

  print(f"Vocab size: {tokenizer.vocab_size}")
  print(f"Output prefix: {args.output_prefix}")
  output_bin_files = {}
  output_idx_files = {}
  builders = {}
  # for key in args.json_keys:
  for key in args.json_keys:
    output_bin_files[key] = "{}_{}_{}.bin".format(args.output_prefix, key,
                                                  level)
    output_idx_files[key] = "{}_{}_{}.idx".format(args.output_prefix, key,
                                                  level)
    builders[key] = indexed_dataset.make_builder(
        output_bin_files[key],
        impl=args.dataset_impl,
        #  dtype=best_fitting_dtype(
        dtype=best_fitting_sft_dtype(tokenizer.vocab_size))

  startup_end = time.time()
  proc_start = time.time()
  total_bytes_processed = 0
  print("Time to startup:", startup_end - startup_start)
  empty_num = 0
  for i, (doc, bytes_processed) in enumerate(encoded_docs, start=1):
    total_bytes_processed += bytes_processed
    for key, sentences in doc.items():
      if len(sentences) == 0:
        empty_num += 1
        continue
      for sentence in sentences:
        builders[key].add_item(torch.IntTensor(sentence))
      builders[key].end_document()
    if i % args.log_interval == 0:
      current = time.time()
      elapsed = current - proc_start
      mbs = total_bytes_processed / elapsed / 1024 / 1024
      print(f"Processed {i} documents",
            f"({i/elapsed} docs/s, {mbs} MB/s).",
            file=sys.stderr)

  for key in args.json_keys:
    print(f'save to {output_idx_files[key]}')
    builders[key].finalize(output_idx_files[key])
  # builders[output_key].finalize(output_idx_files[output_key])
  print(f'empty num: {empty_num}')


if __name__ == '__main__':
  main()
  # python tools/data_process/preprocess_sft_data.py --input chinese_data/bussiness_intent_detect/data.all_business.filter.pydict --output-prefix chinese_data/bussiness_intent_detect/data.all_business.filter  --dataset-impl mmap --tokenizer-type PretrainedFromHF  --append-eod --workers 20 --append-ans --tokenizer-name-or-path bloom-add-ans-tokenizer

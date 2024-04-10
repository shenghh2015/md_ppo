from transformers import AutoTokenizer
import argparse
import os
import multiprocessing
from palframe.nlp import pydict_file_read
import pickle

class Encoder(object):

  def __init__(self, args):
    self.args = args

  def initializer(self):
    # Use Encoder class as a container for global data
    print(self.args.tokenizer_name_or_path)
    Encoder.tokenizer = AutoTokenizer.from_pretrained(
        self.args.tokenizer_name_or_path)

  def encode(self, json_line):
    data = eval(json_line)
    input_ids = Encoder.tokenizer.encode(data['text'])
    return {"input_ids": input_ids}


TOKENIZERS_PARALLELISM = True


def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--input_path", type=str, default=None)
  parser.add_argument("--output_dir", type=str, default=None)
  parser.add_argument("--tokenizer_name_or_path",
                      type=str,
                      default="step1_sft/bloom_add_ans_tokenizer")
  parser.add_argument("--workers", type=int, default=1)
  parser.add_argument("--max_length", type=int, default=1024)
  return parser.parse_args()


def process_data(data, tokenizer, index, total_num, results):
  text = data["text"]
  result = {"input_ids": tokenizer.encode(text), "id": index}
  if index % int(total_num * 0.01) == 0:
    print(f"index: {index}, process {len(results)}")
  return result


def test_tokenizer(tokenizer):
  test_text = "Welcome to Beijing!"
  print(f"test text: {test_text}")
  token_ids = tokenizer.encode(test_text)
  decode_ids = tokenizer.decode(token_ids)
  print(f"token ids: {token_ids}")
  print(f"decoded texts: {decode_ids}")


def main():

  args = get_args()

  # load dataset
  pydict_dataset = pydict_file_read(args.input_path)
  pydict_dataset = list(pydict_dataset)
  print(f"dataset size: {len(pydict_dataset)}")

  # process
  fin = open(args.input_path, 'r', encoding='utf-8')
  encoder = Encoder(args)
  pool = multiprocessing.Pool(args.workers, initializer=encoder.initializer)
  encoded_feats = list(pool.imap(encoder.encode, fin, 25))
  encoded_feats = [
      feat for feat in encoded_feats
      if len(feat) > 0 and len(feat["input_ids"]) < args.max_length
  ]

  # save data
  input_name = os.path.basename(args.input_path)
  output_name = f"{input_name}.ppo_ptx.num.{len(encoded_feats)}.pkl"
  output_path = os.path.join(args.output_dir, output_name)
  print(f"output path: {output_path}")
  with open(output_path, 'wb') as f:
    pickle.dump(encoded_feats, f)


if __name__ == '__main__':
  main()

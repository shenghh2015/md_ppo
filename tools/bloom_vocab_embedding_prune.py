import os.path
import time
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
from tqdm import tqdm


class BloomVocabularyPruner(object):
  def prune(self,
            model_load_path,
            new_tokenizer_name_or_path,
            tokenier_save_path,
            model_save_path,
            make_vocab_size_divisible_by=128,
            overwrite=False):
    # 创建输出目录
    if not os.path.exists(tokenier_save_path):
      os.makedirs(tokenier_save_path)
    assert (not os.path.exists(model_save_path)
            ), f"{model_save_path} already exists"
    os.makedirs(model_save_path)
    # 加载新词表。如果是中文，就是中文的词表
    new_tokenizer = AutoTokenizer.from_pretrained(new_tokenizer_name_or_path)
    # 加载原词表。一般为多语言模型的词表
    old_tokenizer = AutoTokenizer.from_pretrained(model_load_path)

    # 检查新词表是否为原词表的子集
    old_vocab = old_tokenizer.vocab
    new_vocab = new_tokenizer.vocab
    for token in tqdm(new_vocab.keys()):
      if token not in old_vocab:
        raise Exception('{} not exist'.format(token))
    print('new_tokenizer is subset of old_tokenizer')

    add_token_num = (make_vocab_size_divisible_by - \
    (len(new_vocab) % make_vocab_size_divisible_by)) \
    % make_vocab_size_divisible_by

    # 获得新词表中每个token_id到原词表的token_id的映射
    new2old_token_id = {}
    for token, token_id in tqdm(new_vocab.items()):
      old_token_id = old_vocab[token]
      new2old_token_id[token_id] = old_token_id

    model_state_dict = torch.load(
        os.path.join(model_load_path, "pytorch_model_00001-of-00072.bin"))

    old_embddings_weight = model_state_dict['word_embeddings.weight']

    # 对于新词表中的每个token，取出其对应的权重，复制到新模型中
    vocab_size = len(new_tokenizer) + add_token_num
    print(
        f"add {add_token_num} unused tokens to make vocab size divisible "
        f"by {make_vocab_size_divisible_by}, the padded vocab size is {vocab_size}"
    )
    hidden_size = old_embddings_weight.shape[-1]

    new_embeddings = torch.nn.Embedding(vocab_size, hidden_size)
    # 更新词表权重
    self.update_ebeddings(old_embddings_weight, new2old_token_id,
                          new_embeddings, add_token_num)

    model_state_dict['word_embeddings.weight'] = new_embeddings.weight.data

    new_tokenizer.add_tokens(
        [f'[unused{i}]' for i in range(1, add_token_num + 1)])

    # 找两个测试用例检查一下
    self.check(old_embddings_weight, old_tokenizer, new_embeddings.weight.data,
               new_tokenizer, "长风破浪会有时")
    self.check(old_embddings_weight, old_tokenizer, new_embeddings.weight.data,
               new_tokenizer, "今天lunch吃什么")

    new_tokenizer.save_pretrained(tokenier_save_path)
    print(f"new tokenizer has been saved in {tokenier_save_path}")

    print(f"copying old model to {model_save_path}...")
    os.system(f"cp -r {model_load_path}/* {model_save_path}/")
    print(f"copy completed.")
    torch.save(
        model_state_dict,
        os.path.join(model_save_path, "pytorch_model_00001-of-00072.bin"))
    new_tokenizer.save_pretrained(model_save_path)
    print(
        f"old model in {model_load_path} has been pruned and saved in {model_save_path}"
    )
    return

  def update_ebeddings(self, old_embddings_weight, new2old_token_id,
                       new_embeddings, add_token_num):
    for token_id, old_token_id in tqdm(new2old_token_id.items()):
      new_embeddings.weight.data[token_id] = old_embddings_weight[old_token_id]
    if add_token_num > 0:
      new_embeddings.weight.data[-add_token_num:] = 0.0
    return

  def check(self, old_embeddings_weight, old_tokenizer, new_embeddings_weight,
            new_tokenizer, text):
    # sanity check
    old_input_ids = old_tokenizer(text, return_tensors='pt').input_ids
    new_input_ids = new_tokenizer(text, return_tensors='pt').input_ids

    with torch.no_grad():
      old_text_embedding = old_embeddings_weight[old_input_ids]
      new_text_embedding = new_embeddings_weight[new_input_ids]

      diff = (old_text_embedding - new_text_embedding).abs().max()
      assert diff <= 1e-6, "pruning check failed, new embeddings are not consistent with old embeddings"
      print(f"pruning check passed, text: {text}")
    return


def get_args():
  parser = argparse.ArgumentParser()
  group = parser.add_argument_group(title='Bloom-176B model path')
  group.add_argument('--model-load-path',
                     type=str,
                     help='Path to load input models')

  group = parser.add_argument_group(title='tokenizer')
  group.add_argument("--prune-tokenizer-name-or-path",
                     type=str,
                     default="./YeungNLP_bloom_396m_zh_tokenizer",
                     help="Name or path of the huggingface tokenizer.")
  group.add_argument('--make-vocab-size-divisible-by',
                     type=int,
                     default=128,
                     help='Pad the vocab size to be divisible by this value.'
                     'This is added for computational efficieny reasons.')

  group = parser.add_argument_group(title='save path')
  group.add_argument('--tokenizer-save-path',
                     type=str,
                     default="./bloom_pruned_tokenizer/",
                     help='Path for saving pruned tokenizer')
  group.add_argument('--model-save-path',
                     type=str,
                     help='Path for saving pruned model')
  args = parser.parse_args()
  return args


if __name__ == '__main__':
  args = get_args()
  model_load_path = args.model_load_path
  new_tokenizer_name_or_path = args.prune_tokenizer_name_or_path
  tokenizer_save_path = args.tokenizer_save_path
  t0 = time.time()
  pruner = BloomVocabularyPruner()
  # 裁剪
  pruner.prune(model_load_path,
               new_tokenizer_name_or_path,
               tokenizer_save_path,
               model_save_path=args.model_save_path,
               make_vocab_size_divisible_by=args.make_vocab_size_divisible_by)
  t1 = time.time()
  print(f'time consuming is: {t1-t0}')

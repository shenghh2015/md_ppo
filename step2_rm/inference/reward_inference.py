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

import torch

from tqdm import tqdm
from megatron import get_args
from megatron import print_rank_0
from megatron import get_tokenizer
from megatron import mpu, prompt_template
from megatron.checkpointing import load_checkpoint
from megatron.initialize import initialize_megatron
from step2_rm.models.reward_model import GPTModelCritic
from megatron.training import get_model
from megatron.text_generation_utils import recv_forward, unwrap_model, torchDDP, LocalDDP, Float16Module, send_forward, distribute_tokenize, pad_batch

from tools.extral_args import add_step2_train_reward_model_shh_args

reward_eval_data = [
  {'prompt':"中国的首都是哪座城市?",'answer':'北京。'},
  {'prompt':"中国的首都是哪座城市?",'answer':'关你屁事'}
]

def model_provider(args, pre_process=True, post_process=True):
  """Build the model."""

  print_rank_0('building GPT model ...')

  model = GPTModelCritic(num_tokentypes=0,
                         parallel_output=False,
                         pre_process=pre_process,
                         post_process=post_process)

  return model


def get_rw_batch(tokenized_text_batch):
  """Generate batch from context tokens."""
  args = get_args()
  tokenizer = get_tokenizer()
  context_tokens, context_lengths = pad_batch(tokenized_text_batch,
                                              tokenizer.eod, args)
  context_tokens_tensor = torch.cuda.LongTensor(context_tokens)
  context_length_tensor = torch.cuda.LongTensor(context_lengths)

  torch.distributed.broadcast(context_length_tensor,
                              mpu.get_tensor_model_parallel_src_rank(),
                              group=mpu.get_tensor_model_parallel_group())
  torch.distributed.broadcast(context_tokens_tensor,
                              mpu.get_tensor_model_parallel_src_rank(),
                              group=mpu.get_tensor_model_parallel_group())

  tokens = torch.tensor(context_tokens).contiguous().cuda()
  batch_size, maxlen = tokens.shape
  attention_mask = torch.tril(
      torch.ones((batch_size, maxlen, maxlen),
                 device=tokens.device)).view(batch_size, 1, maxlen, maxlen)
  position_ids = torch.arange(maxlen, dtype=torch.long, device=tokens.device)
  position_ids = position_ids.unsqueeze(0).expand_as(tokens)

  attention_mask = (attention_mask < 0.5)

  return tokens, attention_mask, position_ids


def forward_step(
    model,
    tokens,
    position_ids,
    attention_mask,
    tokentype_ids=None,
    layer_past=None,
    get_key_value=None,
):

  # Hidden size changes when not using recompute, need to tell p2p_communicate
  # functions the correct size
  args = get_args()
  orig_seq_length = args.seq_length
  orig_micro_batch_size = args.micro_batch_size
  args.seq_length = tokens.shape[1]
  args.micro_batch_size = tokens.shape[0]

  tokenizer = get_tokenizer()
  input_tensor = recv_forward()

  # Forward pass through the model.
  unwrapped_model = unwrap_model(model, (torchDDP, LocalDDP, Float16Module))
  unwrapped_model.set_input_tensor(input_tensor)

  output_tensor = unwrapped_model.forward_value(tokens,
                                                position_ids,
                                                attention_mask,
                                                tokentype_ids=tokentype_ids,
                                                pad_id=tokenizer.eod)

  if get_key_value:
    output_tensor, layer_past = output_tensor

  send_forward(output_tensor)

  args.seq_length = orig_seq_length
  args.micro_batch_size = orig_micro_batch_size
  if get_key_value:
    return output_tensor, layer_past
  return output_tensor


def generate_reward_model_values(model,
                                 texts,
                                 tokenizer,
                                 eos_token_id,
                                 max_seq_len,
                                 batch_size=None):
  """_summary_

  Args:
      model (_type_): _description_
      prompts (_type_): _description_
      batch_size (int, optional): _description_. Defaults to 2.

  Returns:
      _type_: _description_
  """
  # 支持以batch的形式进行解码
  # print(texts)
  args = get_args()
  if batch_size is None:
    batch_size = args.micro_batch_size
  tokenizer = get_tokenizer()
  if isinstance(texts, str):
    texts = [texts]

  assert len(texts) > 0, texts
  raw_text_lens = [len(text) for text in texts]

  # 分布式tokenizer
  tokenized_texts = distribute_tokenize(texts, tokenizer, args.seq_length)
  assert len(tokenized_texts) == len(texts)

  values = []
  model.eval()
  with torch.no_grad():
    for batch_start in tqdm(range(0, len(tokenized_texts), batch_size),
                            total=len(tokenized_texts) // batch_size):
      torch.cuda.empty_cache()
      tokenized_text_batch = tokenized_texts[batch_start:batch_start +
                                             batch_size]
      tokens, attention_mask, position_ids = get_rw_batch(tokenized_text_batch)
      output_tensor = forward_step(model, tokens, position_ids, attention_mask)
      if mpu.is_pipeline_last_stage() \
        and mpu.get_tensor_model_parallel_rank() == 0:
        chosen_end_scores = [output_tensor['chosen_end_scores'].tolist()]
        # 将数据发送到 rank 0
        torch.distributed.broadcast_object_list(chosen_end_scores,
                                                torch.distributed.get_rank(),
                                                mpu.get_embedding_group())
        values.extend(chosen_end_scores[0])
      elif mpu.is_pipeline_first_stage() \
        and mpu.get_tensor_model_parallel_rank() == 0:
        chosen_end_scores = [None]
        torch.distributed.broadcast_object_list(
            chosen_end_scores, mpu.get_pipeline_model_parallel_last_rank(),
            mpu.get_embedding_group())

        values.extend(chosen_end_scores[0])

    return values


def main():
  """Main program."""

  initialize_megatron(
      extra_args_provider=add_step2_train_reward_model_shh_args,
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

  texts = [
      prompt_template(data['prompt'], data['answer'])
      for data in reward_eval_data
  ]

  tokenizer = get_tokenizer()
  outputs = generate_reward_model_values(model,
                                         texts,
                                         tokenizer,
                                         tokenizer.eod,
                                         args.seq_length,
                                         batch_size=args.micro_batch_size)
  if torch.distributed.get_rank() == 0:
    for i in range(len(texts)):
      print_rank_0(f'text: {texts[i]}, output:{outputs[i]}')


if __name__ == "__main__":
  main()

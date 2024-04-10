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
import torch
from tqdm import tqdm
from megatron import get_args
from megatron import print_rank_0, print_with_rank
from megatron import get_tokenizer
from megatron import mpu, prompt_template
from megatron.checkpointing import load_checkpoint
from megatron.initialize import initialize_megatron
from megatron.model.gpt_model import GPTModel
from megatron.model.gpt_reward_model import GPTModelCritic
from megatron.training import get_model
from megatron.text_generation_utils import generate_and_write_samples_unconditional
from megatron.text_generation_utils import generate_samples_input_from_file
from megatron.text_generation_utils import generate_samples_interactive
from megatron.text_generation_utils import generate_samples
from megatron.text_generation_utils import recv_forward, unwrap_model, torchDDP, LocalDDP, Float16Module, LlamaModel, send_forward, distribute_tokenize, pad_batch
from inference.eval_prompts import ten_prompts, reward_eval_data

from megatron.enums import AttnMaskType
from megatron.model import GPTModel, GPTModelPipe
import deepspeed
from tools.extral_args import add_step2_train_reward_model_args


def model_provider(args, pre_process=True, post_process=True):
  """Build the model."""

  print_rank_0('building GPT model ...')
  # GPTModel.set_input_tensor
  model = GPTModelCritic(num_tokentypes=0,
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
  # Move to GPU.
  # tokens = context_tokens.view(args.micro_batch_size, -1).contiguous().cuda()
  tokens = torch.tensor(context_tokens).contiguous().cuda()

  # Get the attention mask and position ids.
  # attention_mask, _, position_ids = get_ltor_masks_and_position_ids(
  #     tokens,
  #     tokenizer.eod,
  #     False,
  #     False,
  #     args.eod_mask_loss,
  #     prefix_indices=None,
  #     loss_on_targets_only=args.loss_on_targets_only)
  batch_size, maxlen = tokens.shape
  attention_mask = torch.tril(
      torch.ones((batch_size, maxlen, maxlen),
                 device=tokens.device)).view(batch_size, 1, maxlen, maxlen)
  position_ids = torch.arange(maxlen, dtype=torch.long, device=tokens.device)
  position_ids = position_ids.unsqueeze(0).expand_as(tokens)

  attention_mask = (attention_mask < 0.5)
  # attention_mask = None
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
  # print_with_rank('wating recv',tokens.shape)
  input_tensor = recv_forward()
  # print_with_rank('recv')

  # Forward pass through the model.
  unwrapped_model = unwrap_model(model, (torchDDP, LocalDDP, Float16Module))

  # print('sfgsfhg',type(unwrapped_model),type(unwrapped_model.module),dir(unwrapped_model.module))
  # print('sdsfb',unwrapped_model.module.modules())
  # for m in unwrapped_model.module.modules():
  #     print(type(m) )
  # attention_mask = None
  unwrapped_model.set_input_tensor(input_tensor)

  output_tensor = unwrapped_model.forward_value(tokens,
                                                position_ids,
                                                attention_mask,
                                                tokentype_ids=tokentype_ids,
                                                pad_id=tokenizer.eod)

  if get_key_value:
    output_tensor, layer_past = output_tensor

  send_forward(output_tensor)
  # print_with_rank('send')

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
  # print_with_rank(f'tokenized texts is :{tokenized_texts}')
  assert len(tokenized_texts) == len(texts)

  # if dist.get_rank()==0:
  #   print(len(prompts))
  #   for prompt in tokenized_prompts:
  #     print(prompt)

  # batch_num = math.ceil(len(prompts)/batch_size)

  values = []
  model.eval()
  with torch.no_grad():
    for batch_start in tqdm(range(0, len(tokenized_texts), batch_size),
                            total=len(tokenized_texts) // batch_size):
      torch.cuda.empty_cache()
      tokenized_text_batch = tokenized_texts[batch_start:batch_start +
                                             batch_size]
      # raw_text_len_batch = raw_text_lens[batch_start:batch_start+batch_size]
      # tokenizer, 仅仅对于张量并行的第一个位置做,然后不同到其他机器
      # contexts_tensor_len = [len(context_tokens) for context_tokens in tokenized_text_batch]
      # max_context_tensor_len = max(contexts_tensor_len)
      # print_rank_0('max_context_tensor_len',max_context_tensor_len,max_new_tokens,max_seq_len)
      tokens, attention_mask, position_ids = get_rw_batch(tokenized_text_batch)
      # print_with_rank('tokens',tokens)
      output_tensor = forward_step(model, tokens, position_ids, attention_mask)

      # token_stream = get_token_stream(
      #   model,tokenized_prompt_batch,
      #   max_generated_len=min(
      #     max_new_tokens+max_context_tensor_len,max_seq_len
      #     ),
      #   max_seq_len=max_seq_len,
      #   eos_token_id=eos_token_id,
      #   recompute=recompute,
      #   greedy=greedy,
      #   temperature=temperature,
      #   top_k=top_k,
      #   top_p=top_p
      #   )

      # decode_tokens = None
      # for counter, decode_tokens in enumerate(token_stream):
      #   continue

      # decode_tokens, lengths = decode_tokens

      # decode_tokens = decode_tokens[0].tolist()
      if mpu.is_pipeline_last_stage() \
        and mpu.get_tensor_model_parallel_rank() == 0:
        # if decode_tokens is None:
        #   trim_decode_token_list = [None]*len(tokenized_text_batch)
        # else:
        #   trim_decode_token_list = [
        #     tokenizer.detokenize(decode_tokens[i,:lengths[i]])[raw_text_len_batch[i]:]
        #     for i in range(len(tokenized_prompt_batch))
        #   ]
        # chosen_end_scores = [output_tensor['chosen_end_scores'].tolist()]
        chosen_end_scores = [output_tensor['chosen_end_scores'].tolist()]
        # print(f'chosen scores is: {chosen_end_scores}')
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
        # print(trim_decode_token_list)

    return values


def main():
  """Main program."""

  initialize_megatron(extra_args_provider=add_step2_train_reward_model_args,
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
  # Generate samples.
  # if args.num_samples == 0:
  #     args.micro_batch_size = 1
  #     if args.sample_input_file != None:
  #         generate_samples_input_from_file(model)
  #     else:
  #         generate_samples_interactive(model)
  # else:
  #     generate_and_write_samples_unconditional(model)


def main_file():
  import json
  file_path = "rm_data/sunzeyeah_chinese_chatgpt_corpus_300w/回答/sunzeyeah_chinese_chatgpt_corpus_213689_ppo.pydict"
  # model_name = 'bloom-176b-lora-rank-2-3M-data'
  model_name = 'bloom-1b7-reward_model'
  output_path = f"rm_data/sunzeyeah_chinese_chatgpt_corpus_300w/eval_{model_name}.sunzeyeah_chinese_chatgpt_corpus_213689_ppo.pydict"
  save_batch_size = 2000
  # model_name = 'bloom-176b-lora-rank-2-3M-data'
  # model_name = 'bloom-176b-lora-rank-2-3M-data'
  with open(file_path) as f:
    # texts_dicts = [json.loads(line) for line in f.readlines()]
    texts_dicts = [eval(line) for line in f.readlines() if len(line) < 2048]

  initialize_megatron(extra_args_provider=add_step2_train_reward_model_args,
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

  tokenizer = get_tokenizer()
  print_rank_0(f'current prompt num: {len(texts_dicts)}')
  prompt_num = len(texts_dicts)
  # if torch.distributed.get_rank() ==0:
  #   os.unlink(output_path)

  for batch_start_id in range(0, prompt_num, save_batch_size):
    print_rank_0(f'batch_start_id: {batch_start_id}')
    output_dict = {}
    text_data_batch = texts_dicts[batch_start_id:batch_start_id +
                                  save_batch_size]
    chosen_texts = [
        prompt_template(data['prompt'], data['chosen'])
        for data in text_data_batch
    ]
    rejected_texts = [
        prompt_template(data['prompt'], data['rejected'])
        for data in text_data_batch
    ]

    chosen_outputs = generate_reward_model_values(
        model,
        chosen_texts,
        tokenizer,
        tokenizer.eod,
        args.seq_length,
        batch_size=args.micro_batch_size)
    rejected_outputs = generate_reward_model_values(
        model,
        rejected_texts,
        tokenizer,
        tokenizer.eod,
        args.seq_length,
        batch_size=args.micro_batch_size)
    if torch.distributed.get_rank() == 0:
      # 写入到本地
      assert len(chosen_texts) == len(text_data_batch), (len(chosen_outputs),
                                                         len(rejected_outputs))
      for text_data, chosen_output, rejected_output in zip(
          text_data_batch, chosen_outputs, rejected_outputs):
        output_dict['text'] = text_data
        output_dict[f'chosen_score'] = chosen_output
        output_dict[f'rejected_score'] = rejected_output
        with open(output_path, 'a') as f:
          f.write(json.dumps(output_dict, ensure_ascii=False))
          f.write('\n')

  # Generate samples.
  # if args.num_samples == 0:
  #     args.micro_batch_size = 1
  #     if args.sample_input_file != None:
  #         generate_samples_input_from_file(model)
  #     else:
  #         generate_samples_interactive(model)
  # else:
  #     generate_and_write_samples_unconditional(model)


if __name__ == "__main__":
  main()
  # main_file()

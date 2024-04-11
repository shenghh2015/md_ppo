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
"""reward 模型训练"""

from functools import partial
import numpy as np

import pickle
import torch
from functools import partial
from itertools import chain
from torch.nn.utils.rnn import pad_sequence
from datasets import Dataset
from megatron.model.module import Float16Module
from megatron.model.distributed import DistributedDataParallel as LocalDDP
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP
from megatron import get_args
from megatron import print_rank_0
from megatron import get_timers
from megatron import get_tokenizer
from megatron.utils import unwrap_model
from megatron import mpu
from megatron.enums import AttnMaskType
from step2_rm.models.reward_model import GPTModelCritic
from step2_rm.train.training_reward import pretrain
from megatron.utils import get_ltor_masks_and_position_ids
from megatron.utils import average_losses_across_data_parallel_group
from tools.extral_args import add_step2_train_reward_model_args

import deepspeed
from deepspeed.runtime.utils import see_memory_usage
import os

try:
  from torch.distributed.elastic.multiprocessing.errors import record
except ImportError:
  # noop
  def record(fn):
    return fn


def model_provider(pre_process=True, post_process=True):
  """Build the model."""

  print_rank_0('building GPT model ...')
  if torch.distributed.is_initialized(
  ) and not torch.distributed.get_rank() == 0:
    pass
  see_memory_usage(f"Before Building Model", force=True)

  args = get_args()
  
  print(f'reward loss weights: euqal_score {args.equal_score_loss_weight}, 0-positive {args.zero_positive_penalty_weight}, bound {args.reward_bound_penalty_weight}')
  with deepspeed.zero.Init(data_parallel_group=mpu.get_data_parallel_group(),
                           remote_device=None if args.remote_device == 'none'
                           else args.remote_device,
                           config_dict_or_path=args.deepspeed_config,
                           enabled=args.zero_stage == 3,
                           mpu=mpu):
    if args.deepspeed:
      # TODO 支持deepspeed
      args.pretrain_causal_attention = True
      model = GPTModelPipe(num_tokentypes=0,
                           parallel_output=True,
                           attn_mask_type=AttnMaskType.causal)
      # This is a hack to give us a reference to get_batch_pipe from within training.py
      # We need to call model.set_batch_fn after deepspeed.initialize
      model._megatron_batch_fn = get_batch_pipe
    else:
      model = GPTModelCritic(num_tokentypes=0,
                             parallel_output=True,
                             pre_process=pre_process,
                             post_process=post_process,
                             args=args)
  see_memory_usage(f"After Building Model", force=True)
  return model


def get_batch(data_iterator):
  """Generate a batch"""
  args = get_args()
  tokenizer = get_tokenizer()

  # Items and their type.
  # keys = ['text']
  # datatype = torch.int64
  """
  { 
    "prompt": xxx,
    "pairs":[{"chosen":xxx,"reject":xxx},....] ,190
  }
  """

  # Broadcast data.
  if data_iterator is not None:
    data = [next(data_iterator)]

  else:
    data = [None]
  torch.distributed.broadcast_object_list(
      data,
      mpu.get_tensor_model_parallel_src_rank(),
      group=mpu.get_tensor_model_parallel_group())

  data = data[0]
  # print('data_iterator',data_iterator)
  prompt_ids_plus_chosen = [
      datum['prompt_ids'] + datum['chosen_ids'] for datum in data
  ]
  prompt_ids_plus_rejected = [
      datum['prompt_ids'] + datum['rejected_ids'] for datum in data
  ]

  # 按照最大值阶段
  prompt_ids_plus_chosen = [
      i[:args.seq_length] for i in prompt_ids_plus_chosen
  ]
  prompt_ids_plus_rejected = [
      i[:args.seq_length] for i in prompt_ids_plus_rejected
  ]

  # input_lens = [len(i) for i in chain(prompt_ids_plus_chosen,prompt_ids_plus_rejected)]

  inputs_list = [
      torch.LongTensor(i).cuda()
      for i in chain(prompt_ids_plus_chosen, prompt_ids_plus_rejected)
  ]
  input_ids = pad_sequence(inputs_list,
                           batch_first=True,
                           padding_value=tokenizer.pad)

  # score_labels
  labels = [datum['chosen_score'] for datum in data] + [datum['reject_score'] for datum in data]
  score_labels = torch.FloatTensor(labels).cuda()
  # sample_freqs
  freqs = [datum['chosen_freq'] for datum in data] + [datum['reject_freq'] for datum in data]
  sample_freqs = torch.FloatTensor(freqs).cuda()
  
  # print_with_rank(tokenizer.detokenize(input_ids.tolist()[0]))
  # torch.distributed.barrier()

  # print('input_ids',input_ids.dtype)
  #扩展到最大长度
  # if input_ids.shape[1] < args.seq_length:
  #   input_ids = torch.cat(
  #     (
  #       input_ids,
  #       (tokenizer.pad*torch.ones((input_ids.shape[0],args.seq_length-input_ids.shape[1]))).long().cuda()
  #     ),
  #     dim=-1
  #   )
  #
  # print('input_ids',input_ids.dtype)
  batch_size, maxlen = input_ids.shape
  attention_mask = torch.tril(
      torch.ones((batch_size, maxlen, maxlen),
                 device=input_ids.device)).view(batch_size, 1, maxlen, maxlen)
  position_ids = torch.arange(maxlen,
                              dtype=torch.long,
                              device=input_ids.device)
  position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

  attention_mask = (attention_mask < 0.5)

  # print(score_labels)
  # print(sample_freqs)
  return input_ids, attention_mask, position_ids, score_labels, sample_freqs


def get_batch_pipe(data):
  """Modification of `get_batch` to work on `next(data_iterator)` instead of `data_iterator`"""
  args = get_args()
  tokenizer = get_tokenizer()

  # Items and their type.
  keys = ['text']
  datatype = torch.int64

  # Broadcast data.
  data_b = mpu.broadcast_data(keys, data, datatype)

  # Unpack.
  tokens_ = data_b['text'].long()
  labels = tokens_[:, 1:].contiguous()
  tokens = tokens_[:, :-1].contiguous()

  # Get the masks and position ids.
  attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
      tokens,
      tokenizer.eod,
      args.reset_position_ids,
      args.reset_attention_mask,
      args.eod_mask_loss,
      prefix_indices=None,
      loss_on_targets_only=args.loss_on_targets_only)

  print_rank_0(f'loss_mask_sum: {loss_mask.sum(dim=-1)}')
  return (tokens, position_ids, attention_mask), (labels, loss_mask)


def loss_func(loss, training=True):
  # losses = output_tensor.float()
  # loss_mask = loss_mask.view(-1).float()
  # loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

  # Reduce loss for logging.
  # print('loss',loss.shape,loss)
  if training:
    averaged_loss = average_losses_across_data_parallel_group([loss])
    return loss, {'lm loss': averaged_loss[0]}
  else:
    eval_loss = loss['loss']
    chosen_end_scores = loss['chosen_end_scores']
    reject_end_scores = loss['reject_end_scores']

    # avaeraged_eval_loss = average_losses_across_data_parallel_group([eval_loss])[0]
    avaeraged_eval_loss_output = [None] * mpu.get_data_parallel_world_size()
    chosen_end_scores_output = [None] * mpu.get_data_parallel_world_size()
    reject_end_scores_output = [None] * mpu.get_data_parallel_world_size()

    # 同步预测结果

    torch.distributed.all_gather_object(avaeraged_eval_loss_output,
                                        eval_loss,
                                        group=mpu.get_data_parallel_group())

    torch.distributed.all_gather_object(chosen_end_scores_output,
                                        chosen_end_scores,
                                        group=mpu.get_data_parallel_group())

    torch.distributed.all_gather_object(reject_end_scores_output,
                                        reject_end_scores,
                                        group=mpu.get_data_parallel_group())

    avaeraged_eval_loss = sum(avaeraged_eval_loss_output) / len(
        avaeraged_eval_loss_output)
    chosen_end_scores = sum(chosen_end_scores_output, [])
    reject_end_scores = sum(reject_end_scores_output, [])

    return avaeraged_eval_loss, {
        "loss": avaeraged_eval_loss,
        "chosen_end_scores": chosen_end_scores,
        "reject_end_scores": reject_end_scores
    }


def forward_step(data_iterator, model, training=True):
  """Forward step."""
  args = get_args()
  timers = get_timers()
  tokenizer = get_tokenizer()

  # Get the batch.
  timers('batch-generator').start()
  input_ids, attention_mask, position_ids, score_labels, sample_freqs = get_batch(data_iterator)
  timers('batch-generator').stop()
  # print('max input_ids',input_ids.max())
  if training:
    output_tensor = model(input_ids,
                          position_ids,
                          attention_mask,
                          pad_id=tokenizer.pad,
                          reward_model_training=True,
                          score_labels=score_labels,
                          sample_freqs=sample_freqs)
  else:
    # 评估状态的forward
    unwrapped_model = unwrap_model(model, (torchDDP, LocalDDP, Float16Module))

    output_tensor = unwrapped_model.forward_eval(input_ids,
                                                 position_ids,
                                                 attention_mask,
                                                 pad_id=tokenizer.pad,
                                                 prompt_length=0,
                                                 score_labels=score_labels,
                                                 sample_freqs=sample_freqs)

  return output_tensor, partial(loss_func, training=training)


class FixLengthDataset(torch.utils.data.Dataset):
  """
  如果数据索引超过数据集长度, 就行取模操作, 保证总是可以取到数据。
  同时又拥有一个总长度
  Args:
      torch (_type_): _description_
  """
  def __init__(self, dataset, total_sample_mum) -> None:
    super().__init__()
    # assert total_sample_mum >= len(dataset), f"total_sample_mum: {total_sample_mum}, dataset num: {len(dataset)} "
    self.total_sample_num = total_sample_mum
    self.dataset = dataset
    self.dataset_len = len(dataset)

  def __getitem__(self, idx):
    # print('getitem',idx)
    # print(self.dataset[idx % self.dataset_len])
    return self.dataset[idx % self.dataset_len]

  def __len__(self):
    return self.total_sample_num


def train_valid_test_datasets_provider(train_val_test_num_samples):
  """Build train, valid, and test datasets."""
  args = get_args()
  train_ds, valid_ds, test_ds = None, None, None
  train_samples, valid_samples, test_samples = train_val_test_num_samples

  print_rank_0('> building train, validation, and test datasets for GPT ...')

  # Option 1 of data loading using --data-path
  def load_pydict(path):
    with open(path, 'rb') as f:
      data = pickle.load(f)
      # print(len(data))
      # print(data[0])
      yield from iter(data)

  # dataset = Dataset.from_generator(partial(load_pydict,args.data_path[0]))
  # ds = dataset.train_test_split(test_size=0.2,generator=np.random.default_rng(args.seed))
  # train_ds, valid_ds, test_ds = ds['train'],ds['test'],ds['test']

  print(args.data_path[0])
  print(args.test_data_path)
  train_dataset = Dataset.from_generator(
      partial(load_pydict, args.data_path[0]))
  test_dataset = Dataset.from_generator(
      partial(load_pydict, args.test_data_path))
  # shuffle samples to make the data from the same prompt evenly distributed
  train_dataset = train_dataset.shuffle(args.seed)
  test_dataset = test_dataset.shuffle(args.seed)
  train_ds, valid_ds, test_ds = train_dataset, test_dataset, test_dataset

  # print(len(train_ds),train_ds[0])

  train_ds = FixLengthDataset(train_ds, train_samples)
  valid_ds = FixLengthDataset(valid_ds, valid_samples)
  test_ds = FixLengthDataset(test_ds, test_samples)

  return train_ds, valid_ds, test_ds


@record
def main():
  pretrain(train_valid_test_datasets_provider,
           model_provider,
           forward_step,
           collate_fn=lambda x: x,
           extra_args_provider=add_step2_train_reward_model_args)


if __name__ == "__main__":
  main()

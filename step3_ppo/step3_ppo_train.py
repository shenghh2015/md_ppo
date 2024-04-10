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
"""Distributed PPO training with megatron """

import os
import sys
import time
import pickle
import random
from functools import partial
from transformers import get_constant_schedule_with_warmup
from megatron.model.distributed import DistributedDataParallel as LocalDDP
from megatron.model.module import Float16Module
from megatron.utils import unwrap_model
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP
from megatron import prompt_template

sys.path.append('/root/.cache/torch_extensions')
my_env = os.environ.copy()
my_env["PATH"] = "/opt/conda/bin:" + my_env["PATH"]
my_env['CUDA_HOME'] = '/usr/local/cuda'
os.environ.update(my_env)

import torch
from megatron import get_args
from megatron import print_rank_0, print_with_rank
from megatron import get_timers
from megatron import get_tokenizer
from megatron import mpu
from megatron.training_ppo_with_actor_critic import ppo_train

import deepspeed
from megatron.model.gpt_model_ppo import ActorCriticModel
from deepspeed.runtime.utils import see_memory_usage
import os, copy
import numpy as np
import torch.distributed as dist
try:
  from torch.distributed.elastic.multiprocessing.errors import record
except ImportError:
  # noop
  def record(fn):
    return fn


import requests
from step2_reward_model.rules.rule_matcher import detect_abnormal_generation

# model size map
model_size_map = {
    '1b7': {
        "num_layers": 24,
        "hidden_size": 2048,
        "num_attention_heads": 16
    },
    "7b1": {
        "num_layers": 30,
        "hidden_size": 4096,
        "num_attention_heads": 32
    },
    "176b": {
        "num_layers": 70,
        "hidden_size": 14336,
        "num_attention_heads": 112
    }
}


class ListDataset(torch.utils.data.Dataset):

  def __init__(self, samples) -> None:
    super().__init__()
    self.samples = samples

  def __getitem__(self, i):
    return self.samples[i % len(self)]

  def __len__(self):
    return len(self.samples)


class AdaptiveKLController:

  def __init__(self, init_kl_coef, target_kl, k_beta=0.1):

    self.value = init_kl_coef
    self.target_kl = target_kl
    self.k_beta = k_beta

  def update(self, current_kl: float, step=None):
    proportional_error = np.clip(current_kl / self.target_kl - 1, -0.2, 0.2)
    mult = 1.0 + self.k_beta * proportional_error
    self.value *= mult


class FixedKLController:
  """Fixed KL controller."""

  def __init__(self, kl_coef):
    self.value = kl_coef

  def update(self, current: float = None, n_steps: int = None):
    """Returns updated KL coefficient, βₜ₊₁.
        Arguments:
            current: The current KL value between the newest policy and the initial policy.
        """
    pass


# actor-critic model provider
def model_provider(pre_process=True, post_process=True):
  """Build the model (ActirCritic)."""

  print_rank_0('building ActirCritic model ...')
  if torch.distributed.is_initialized(
  ) and not torch.distributed.get_rank() == 0:
    pass
  see_memory_usage(f"Before Building Model", force=True)

  args = get_args()

  with deepspeed.zero.Init(data_parallel_group=mpu.get_data_parallel_group(),
                           remote_device=None if args.remote_device == 'none'
                           else args.remote_device,
                           config_dict_or_path=args.deepspeed_config,
                           enabled=args.zero_stage == 3,
                           mpu=mpu):

    # actor and critic args
    actor_args = copy.deepcopy(args)
    critic_args = copy.deepcopy(args)

    # define actor model param
    actor_args.num_layers = model_size_map[args.actor_model]['num_layers']
    actor_args.hidden_size = model_size_map[args.actor_model]['hidden_size']
    actor_args.num_attention_heads = model_size_map[
        args.actor_model]['num_attention_heads']
    actor_args.custom_layer_split = args.actor_custom_layer_split

    args.num_layers = actor_args.num_layers
    args.hidden_size = actor_args.hidden_size
    args.num_attention_heads = actor_args.num_attention_heads

    # define critic model param
    critic_args.num_layers = model_size_map[args.critic_model]['num_layers']
    critic_args.hidden_size = model_size_map[args.critic_model]['hidden_size']
    critic_args.num_attention_heads = model_size_map[
        args.critic_model]['num_attention_heads']
    critic_args.custom_layer_split = args.critic_custom_layer_split

    # build actor and critic model
    model = ActorCriticModel(actor_args,
                             critic_args,
                             args,
                             num_tokentypes=0,
                             parallel_output=True,
                             pre_process=pre_process,
                             post_process=post_process)

  see_memory_usage(f"After Building Model", force=True)

  return model


# group learning rate provider
def param_groups_provider(modules):
  """ Group learning rate provider """

  args = get_args()

  weight_decay_params = []
  no_weight_decay_params = []
  for module in modules:
    for n, v in module.named_parameters():
      if 'layernorm' in n or 'bias' in n:
        no_weight_decay_params.append((n, v))
      else:
        weight_decay_params.append((n, v))

  # set smaller lr to actor and larger one to critic
  no_weight_decay_params_actor = [
      v for n, v in no_weight_decay_params if 'actor' in n
  ]
  no_weight_decay_params_critic = [
      v for n, v in no_weight_decay_params if 'actor' not in n
  ]
  weight_decay_params_actor = [
      v for n, v in weight_decay_params if 'actor' in n
  ]
  weight_decay_params_critic = [
      v for n, v in weight_decay_params if 'actor' not in n
  ]
  param_groups = [
      {
          "params": no_weight_decay_params_actor,
          'lr': args.actor_lr,
          'weight_decay': 0.0
      },
      {
          "params": no_weight_decay_params_critic,
          'lr': args.critic_lr,
          'weight_decay': 0.0
      },
      {
          "params": weight_decay_params_actor,
          'lr': args.actor_lr
      },
      {
          "params": weight_decay_params_critic,
          'lr': args.critic_lr
      },
  ]

  if mpu.is_pipeline_last_stage():
    for module in modules:
      print(module)
    print('no_weight_decay_params_actor', len(no_weight_decay_params_actor))
    print('no_weight_decay_params_critic', len(no_weight_decay_params_critic))
    print('weight_decay_params_actor', len(weight_decay_params_actor))
    print('weight_decay_params_critic', len(weight_decay_params_critic))
    for group in param_groups:
      print('group lr', len(group['params']), group['lr'])

  return param_groups


def learning_rate_scheduler_provider(optimizer):
  args = get_args()
  lr_schedule = get_constant_schedule_with_warmup(optimizer.optimizer,
                                                  args.ppo_lr_num_warmup_steps)
  lr_schedule.step()
  return lr_schedule


# brachcast batch data while optimizing the actor and critic
def brocast_batch_data(data: list):
  if data is not None:
    assert isinstance(data, list)
    data = [[{k: v for k, v in datum.items()} for datum in data]]
  else:
    data = [None]
  tmp_src_rank = mpu.get_tensor_model_parallel_src_rank()
  dist.broadcast_object_list(data,
                             tmp_src_rank,
                             group=mpu.get_tensor_model_parallel_group())
  data = data[0]
  data = [{k: v for k, v in datum.items()} for datum in data]
  return data


# batch of data for optimizing the actor and critic
def get_batch(data_iterator):
  """Generate a batch"""

  tokenizer = get_tokenizer()  # tokenizer

  # Broadcast data.
  if data_iterator is not None:
    data = next(data_iterator)
  else:
    data = None

  data_b = brocast_batch_data(data)
  max_token_len = max([datum['input_tokens'].shape[1] for datum in data_b])

  # padding
  tokens_list = []
  for datum_b in data_b:
    tokens_ = datum_b['input_tokens']
    tokens_ = torch.cat(
        (tokens_,
         torch.LongTensor([[tokenizer.eod] *
                           (max_token_len - tokens_.shape[1])])),
        dim=-1)
    tokens_list.append(tokens_)

  # batch size
  tokens = torch.cat(tokens_list, dim=0).long()
  micro_batch_size, seq_length = tokens.size()

  # causal mask: (micro_batch_size, seq_len, seq_len)
  attention_mask = torch.tril(
      torch.ones((micro_batch_size, seq_length,
                  seq_length))).view(micro_batch_size, 1, seq_length,
                                     seq_length)
  attention_mask = (attention_mask < 0.5)

  # position ids
  position_ids = torch.arange(seq_length, dtype=torch.long)
  position_ids = position_ids.unsqueeze(0).expand_as(tokens)

  # padding function
  def _pad(tensor_list, maxlen, pad_value):
    tensor_list = [
        torch.cat(
            (tensor, torch.tensor([[pad_value] * (maxlen - tensor.shape[1])])),
            dim=-1) for tensor in tensor_list
    ]
    return torch.cat(tensor_list, dim=0)

  # input data to cuda
  tokens = tokens.cuda()
  attention_mask = attention_mask.cuda()
  position_ids = position_ids.cuda()

  if mpu.is_pipeline_last_stage():
    query_tensors_list = [data['query_tensors'] for data in data_b]
    query_tensor_len = torch.LongTensor(
        [query_tensors.shape[1] for query_tensors in query_tensors_list])
    old_rewards_list = [data['old_rewards'] for data in data_b]
    old_values_list = [data['old_values'] for data in data_b]
    old_logprobs_list = [data['old_logprobs'] for data in data_b]
    batch_size = len(query_tensors_list)
    response_lengths = [
        old_logprobs_list[i].shape[1] for i in range(batch_size)
    ]
    max_response_length = max(response_lengths)

    # old_rewards, old_values, and old_logprobs: (batch, max_response_length)
    old_rewards = _pad(old_rewards_list, max_response_length, 0.0)
    old_values = _pad(old_values_list, max_response_length, 0.0)
    old_logprobs = _pad(old_logprobs_list, max_response_length, 0.0)

    # response_mask
    response_mask = torch.arange(max_response_length)[
        None, :] < torch.LongTensor(response_lengths)[:, None]

    # data to cuda
    old_logprobs = old_logprobs.cuda()
    old_values = old_values.cuda()
    old_rewards = old_rewards.cuda()
    query_tensor_len = query_tensor_len.cuda()
    response_mask = response_mask.cuda()

    # return more complete information from in the last pipeline stage
    return {
        'tokens': tokens,
        'attention_mask': attention_mask,
        'position_ids': position_ids,
        'old_logprobs': old_logprobs,
        'old_values': old_values,
        'old_rewards': old_rewards,
        'query_tensors_len': query_tensor_len,
        'response_mask': response_mask
    }

  # return only the input data before the last pipeline stage
  return {
      'tokens': tokens,
      'attention_mask': attention_mask,
      'position_ids': position_ids
  }


# loss function
def loss_func(output_tensor):
  loss, loss_stat = output_tensor
  return loss, loss_stat


# Forward step in actor and critic training
def forward_step(data_iterator, model):
  """Forward step for PPO """

  timers = get_timers()

  # Get a batch.
  timers('batch-generator').start()
  batch_data = get_batch(data_iterator)

  # Get input data (cuda)
  tokens = batch_data['tokens']
  attention_mask = batch_data['attention_mask']
  position_ids = batch_data['position_ids']

  # Get the old_logprobs, old_values, old_rewards, response_mask, and query_tensors_len (cuda)
  # It is only visible to the last pipeline stage.
  old_logprobs = batch_data.get('old_logprobs', None)
  old_values = batch_data.get('old_values', None)
  old_rewards = batch_data.get('old_rewards', None)
  response_mask = batch_data.get('response_mask', None)
  query_tensors_len = batch_data.get('query_tensors_len', None)

  timers('batch-generator').stop()

  # Forward the got batch data to the actor and critic model to compute the output and loss
  output_tensor = model(tokens,
                        position_ids,
                        attention_mask,
                        old_logprobs=old_logprobs,
                        old_values=old_values,
                        old_rewards=old_rewards,
                        response_mask=response_mask,
                        query_tensors_len=query_tensors_len)

  # Return output and loss
  return output_tensor, loss_func


def ptx_forward_step(data_iterator, model):
  """ Forward step for pretraining """

  timers = get_timers()
  args = get_args()
  tokenizer = get_tokenizer()
  timers('batch-generator').start()

  # Broadcast data.
  if data_iterator is not None:
    data = next(data_iterator)
  else:
    data = None

  data_b = brocast_batch_data(data)
  max_token_len = min(max([len(datum['input_ids']) for datum in data_b]),
                      args.seq_length)

  # pad the input data
  tokens_list = []
  for datum_b in data_b:
    tokens_ = torch.LongTensor([datum_b['input_ids'][:args.seq_length]])
    tokens_ = torch.cat(
        (tokens_,
         torch.LongTensor([[tokenizer.pad] *
                           (max_token_len - tokens_.shape[1])])),
        dim=-1)
    tokens_list.append(tokens_)

  # token and labels for next token prediction
  tokens = torch.cat(tokens_list, dim=0).long()
  tokens, labels = tokens[:, :-1], tokens[:, 1:]

  # causal mask
  micro_batch_size, seq_length = tokens.size()
  attention_mask = torch.tril(
      torch.ones((micro_batch_size, seq_length,
                  seq_length))).view(micro_batch_size, 1, seq_length,
                                     seq_length)
  attention_mask = (attention_mask < 0.5)

  # position ids
  position_ids = torch.arange(seq_length, dtype=torch.long)
  position_ids = position_ids.unsqueeze(0).expand_as(tokens)

  # input data to cuda
  tokens = tokens.cuda()
  labels = labels.cuda()
  attention_mask = attention_mask.cuda()
  position_ids = position_ids.cuda()

  # loss mask
  loss_mask = torch.ones(tokens.size(),
                         dtype=torch.float,
                         device=tokens.device)
  loss_mask[tokens == tokenizer.pad] = 0.0

  timers('batch-generator').stop()

  # loss function
  def _loss_fn(loss_mask, output_tensor):

    assert mpu.is_pipeline_last_stage()

    # loss calculation
    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()
    loss = loss * args.ptx_coeff
    return loss, {'losses/ptx_loss': loss.item()}

  unwrapped_model = unwrap_model(model, (torchDDP, LocalDDP, Float16Module))

  # forward to the actor model to compute the ptx loss
  output_tensor = unwrapped_model.actor(tokens,
                                        position_ids,
                                        attention_mask,
                                        labels=labels)

  return output_tensor, partial(_loss_fn, loss_mask)


def kl_ctl_provider(ppo_config):
  if ppo_config['adap_kl_ctrl']:
    kl_ctl = AdaptiveKLController(
        ppo_config['init_kl_coef'],
        ppo_config['target'],
        ppo_config['k_beta'],
    )
  else:
    kl_ctl = FixedKLController(ppo_config['init_kl_coef'])
  return kl_ctl


def prompts_dataset_provider():
  """ Train/eval prompt dataset provider """

  args = get_args()

  # remove redundant prompts
  def _remove_redundant_prompts(prompts):
    _unique_prompts = set()
    for prompt in prompts:
      if not prompt in _unique_prompts:
        _unique_prompts.add(prompt)
    return list(_unique_prompts)

  # debug flag used to shrink the dataset size for faster debugging
  debug = False

  # load ppo train prompts
  prompts_dataset_path = args.prompts_dataset_path
  prompts = [
      prompt_template(eval(line)['conversations'][0]['value'])
      for line in open(prompts_dataset_path)
  ]
  prompts = _remove_redundant_prompts(prompts)
  if debug: prompts = prompts[:30]

  # tokenize the train prompts
  max_token_len_in_prompt = args.seq_length
  tokenizer = get_tokenizer()
  lens = [len(tokenizer.tokenize(prompt)) for prompt in prompts]
  prompts = [
      prompt for i, prompt in enumerate(prompts)
      if lens[i] < max_token_len_in_prompt
  ]
  random.Random(args.seed).shuffle(prompts)
  # print statitics of training prompts
  print_rank_0(
      f'>> token len stats: mean: {np.mean(lens)}, min: {min(lens)}, max: {max(lens)}'
  )
  print_with_rank(
      f'>> 当前ppo数据集:{prompts_dataset_path}, prompts num:{len(prompts)}')
  prompt_dataset = ListDataset(prompts)

  # ppo evaluation prompts
  eval_prompts_dataset_path = args.eval_prompts_dataset_path
  if eval_prompts_dataset_path is not None:
    eval_prompts = [
        prompt_template(eval(line)['conversations'][0]['value'])
        for line in open(eval_prompts_dataset_path)
    ]
    eval_lens = [len(tokenizer.tokenize(prompt)) for prompt in eval_prompts]
    if debug: eval_prompts = eval_prompts[:10]
    print_rank_0('> eval prompts: ', eval_prompts[:3])
    print_rank_0(
        f'>> token len stats: mean: {np.mean(eval_lens)}, min: {min(eval_lens)}, max: {max(eval_lens)}'
    )
  else:
    eval_prompts = None

  return prompt_dataset, eval_prompts


def ptx_dataset_provider():
  """ Pretraining dataset provider """

  args = get_args()
  ptx_datapath = args.ptx_dataset_path
  with open(ptx_datapath, 'rb') as f:
    data = pickle.load(f)

  print_with_rank(f'当前预训练数据集: {ptx_datapath}, prompts num: {len(data)}')
  return ListDataset(data)


def reward_fn(texts):
  """ Reward function API """

  args = get_args()
  assert isinstance(texts, list) and isinstance(texts[0], str), texts

  # get reward scores from a RM inference flask api
  ppo_reward_model_address = args.ppo_reward_model_address
  t0 = time.time()
  r = requests.post(ppo_reward_model_address,
                    json={
                        'user_input': texts,
                        'use_prompt_template': False
                    })
  assert r.status_code == 200, r.text
  r = r.json()
  rewards = r['rewards']
  print_rank_0('reward model exe time', r['model_exe_time'], 'api time',
               time.time() - t0)

  # rule-based rejection
  if args.use_rule_based_reward:
    for i in range(len(texts)):
      if detect_abnormal_generation(texts[i]):
        rewards[i] = args.rule_based_min_value
  return rewards


def ref_model_fn(data):
  """ Reference model API """

  args = get_args()
  ppo_ref_model_address = args.ppo_ref_model_address
  t0 = time.time()
  r = requests.post(ppo_ref_model_address, json=data)
  assert r.status_code == 200, r.text
  r = r.json()
  lm_logtis_list = r['lm_logtis']
  print('ref model exe time', r['model_exe_time'], 'api time',
        time.time() - t0)

  lm_logtis_list = [torch.tensor(lm_logtis) for lm_logtis in lm_logtis_list]

  return lm_logtis_list


# ppo args provider
def extra_args_provider(parser):

  def _add_ppo_args(parser):
    group = parser.add_argument_group('ppo_config', 'ppo config')

    group.add_argument('--prompts_dataset_path',
                       default=None,
                       help='prompts dataset path')

    group.add_argument('--eval_prompts_dataset_path',
                       default=None,
                       help='eval prompts dataset path')

    group.add_argument('--eval_interval_iteration',
                       default=1,
                       type=int,
                       help='eval interval iteration, default is 10')

    group.add_argument('--actor_model',
                       default=None,
                       choices=['1b7', '7b1', '176b'],
                       help="actor model size")

    group.add_argument('--critic_model',
                       default=None,
                       choices=['1b7', '7b1', '176b'],
                       help="critic model size")

    group.add_argument('--actor_lr',
                       default=9.65e-6,
                       type=float,
                       help="actor model lr")

    group.add_argument('--critic_lr',
                       default=5e-6,
                       type=float,
                       help="critic  model lr")

    group.add_argument('--use-v-head-layernorm',
                       action='store_true',
                       help='use v-head layernorm to calculate')

    group.add_argument("--ptx_dataset_path",
                       default=None,
                       help="dataset path for pretraining data")

    group.add_argument('--ptx_coeff',
                       type=float,
                       default=16.0,
                       help='The coefficient for the ptx loss.')

    group.add_argument('--ppo_config_file',
                       default=None,
                       help='ppo config path')

    # flask APIs for ref model and reward model
    group.add_argument('--ppo_ref_model_address',
                       default=None,
                       help='ppo ref model address')
    group.add_argument('--ppo_reward_model_address',
                       default=None,
                       help='ppo reward model address')

    group.add_argument('--ppo_generate_microbatches',
                       type=int,
                       default=1,
                       help="ppo generate pipeline microbatches num")

    group.add_argument('--ppo_generate_batch_size',
                       type=int,
                       default=None,
                       help='ppo generate batch size')

    group.add_argument('--ppo_sampling_times',
                       type=int,
                       default=1,
                       help='number of responses per prompts in ppo sampling')

    group.add_argument('--ppo_lr_num_warmup_steps',
                       type=int,
                       default=10,
                       help='ppo lr num warmup steps')

    group.add_argument('--actor-custom-layer-split',
                       type=lambda x: eval(x),
                       default=None,
                       help='')

    group.add_argument('--critic-custom-layer-split',
                       type=lambda x: eval(x),
                       default=None,
                       help='')

    group.add_argument('--use_rule_based_reward',
                       action='store_true',
                       help='use rule-based method or not')

    group.add_argument('--rule_based_min_value',
                       type=float,
                       default=-6.0,
                       help='')

    return parser

  return _add_ppo_args(parser)


@record
def main():
  ppo_train(prompts_dataset_provider,
            model_provider,
            forward_step,
            reward_fn,
            ref_model_fn,
            kl_ctl_provider,
            extra_args_provider=extra_args_provider,
            param_groups_provider=param_groups_provider,
            learning_rate_scheduler_provider=learning_rate_scheduler_provider,
            ptx_dataset_provider=ptx_dataset_provider,
            ptx_forward_step_func=ptx_forward_step)


if __name__ == "__main__":
  main()

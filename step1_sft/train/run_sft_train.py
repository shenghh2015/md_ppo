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

"""Pretrain GPT"""

import torch
from functools import partial
from megatron import get_args
from megatron import print_rank_0,print_with_rank
from megatron import get_timers
from megatron import get_tokenizer
from megatron import tensor_empty_check
from tools.extral_args import add_step1_sft_args
from megatron import mpu
from megatron.data.gpt_dataset import build_train_valid_test_datasets, build_dataset_group
from megatron.enums import AttnMaskType
from megatron.model import GPTModel, GPTModelPipe
from megatron.training import pretrain
from megatron.utils import get_sft_ltor_masks_and_position_ids
from megatron.utils import average_losses_across_data_parallel_group
from megatron.model.fused_layer_norm import MixedFusedLayerNorm as LayerNorm

import deepspeed
from datasets import Dataset
from deepspeed.runtime.utils import see_memory_usage
from deepspeed.runtime import utils as ds_utils
import sys
import pickle
import torch.distributed as dist
from deepspeed.runtime.pipe.engine import PipelineEngine
from deepspeed.runtime.pipe.engine import schedule
from torch.nn.utils.rnn import pad_sequence

sys.path.append('/root/.cache/torch_extensions')

try:
    from torch.distributed.elastic.multiprocessing.errors import record
except ImportError:
    # noop
    def record(fn):
        return fn


def _exec_reduce_tied_grads(self):
  # We need to run this first to write to self.averaged_gradients;
  # since this class turns `enable_backward_allreduce` off,
  # `self.overlapping_partition_gradients_reduce_epilogue()` defined in the DeepSpeedEngine
  # never actually runs. I suspect this is because of efficiency problems; get_flat_partition in
  # stage2.py might do something expensive; someone will have to look into that later. But
  # in the meantime, this fixes ZeRO2 + Pipelining enough to run a demo. Further profiling
  # needed to decide if it actually breaks everything.
  # (see https://github.com/EleutherAI/gpt-neox/issues/62#issuecomment-761471944)
  if self.zero_optimization_partition_gradients():
    self.optimizer.overlapping_partition_gradients_reduce_epilogue()

  weight_group_list = self.module.get_tied_weights_and_groups()
  for weight, group in weight_group_list:
    grad = weight._hp_grad if self.bfloat16_enabled() else weight.grad
    if grad is not None:
      # print_with_rank('all_reduceing tied grad')
      dist.all_reduce(grad, group=group)
      # print_with_rank('all_reduced tied grad')

PipelineEngine._INSTRUCTION_MAP[schedule.ReduceTiedGrads] = _exec_reduce_tied_grads

def partition_uniform(num_items, num_parts, first_stage_layer_num=3, last_stage_layer_num=5):
  args = get_args()
  if args.custom_deepspeed_layer_split is None:
    return old_partition_uniform(num_items, num_parts)
  custom_deepspeed_layer_split = args.custom_deepspeed_layer_split
  assert isinstance(custom_deepspeed_layer_split,list),type(custom_deepspeed_layer_split)
  if isinstance(custom_deepspeed_layer_split[0], list):
    custom_deepspeed_layer_split = sum(custom_deepspeed_layer_split, []) 
  assert len(custom_deepspeed_layer_split) == num_parts, (num_parts,custom_deepspeed_layer_split)
  assert sum(custom_deepspeed_layer_split) == num_items, (num_items,custom_deepspeed_layer_split)
  parts = [0] * (num_parts + 1)
  offset = 0
  for p in range(1, num_parts+1):
    parts[p] = offset + custom_deepspeed_layer_split[p-1]
    offset = parts[p]
  return parts
old_partition_uniform = ds_utils.partition_uniform
ds_utils.partition_uniform = partition_uniform

def model_provider(pre_process=True, post_process=True):
    """Build the model."""

    print_rank_0('building GPT model ...')
    see_memory_usage(f"Before Building Model", force=True)

    args = get_args()

    with deepspeed.zero.Init(data_parallel_group=mpu.get_data_parallel_group(),
                             remote_device=None if args.remote_device == 'none' else args.remote_device,
                             config_dict_or_path=args.deepspeed_config,
                             enabled=args.zero_stage == 3,
                             mpu=mpu):
        if args.deepspeed:
            print_rank_0('model is gptmodelpipe')
            args.pretrain_causal_attention = False
            model = GPTModelPipe(
                num_tokentypes=0,
                parallel_output=True,
                attn_mask_type=AttnMaskType.custom
            )
            # This is a hack to give us a reference to get_batch_pipe from within training.py
            # We need to call model.set_batch_fn after deepspeed.initialize
            model._megatron_batch_fn = get_batch_pipe
        else:
            print_rank_0('model is gptmodel')
            model = GPTModel(
                num_tokentypes=0,
                parallel_output=True,
                pre_process=pre_process,
                post_process=post_process
            )
    see_memory_usage(f"After Building Model", force=True)
    return model


def __get_batch(data_iterator):
    """Generate a batch"""
    # print("length", len(data_iterator))
    args = get_args()
    tokenizer = get_tokenizer()
    # Items and their type.
    keys = ['text']
    datatype = torch.int64

    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    data_b = mpu.broadcast_data(keys, data, datatype)

    # Unpack.
    tokens_ = data_b['text'].long()
    labels = tokens_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()
    prefix_indices = []
    for b in range(tokens.shape[0]):
        # Find indecies where EOD token is.
        eod_index = (tokens[b] == tokenizer.eod).nonzero().squeeze()
        if (tokens[b] == tokenizer.eod).sum() == 1:
            print('shape error', tokens[b].shape,'eod shape', eod_index, eod_index.shape)
            eod_index = torch.tensor([eod_index, tokens.shape[1]])
        elif (tokens[b] == tokenizer.eod).sum() == 0:
            print('shape error', tokens[b].shape,'eod shape', eod_index, eod_index.shape)
            eod_index = torch.tensor([0, tokens.shape[1]])
        new_tokens = tokens[b, eod_index[0]+1:eod_index[-1]+1]
        # 删除ans并找到新的prefix位置，ans的位置就是answer开始的位置
        ans_index = (new_tokens == tokenizer.ans).nonzero().squeeze()
        mask_wo_ans = new_tokens != tokenizer.ans
        tokens_wo_ans = new_tokens[mask_wo_ans]
        # print(f'tokens wo ans "{tokenizer.detokenize(tokens_wo_ans.tolist())[:100]}, shape is {tokens_wo_ans.shape}')
        if (new_tokens == tokenizer.ans).sum() != 0:
            prefix_index = ans_index - torch.arange(ans_index.shape[0], device=tokens.device)
        else: 
            prefix_index = torch.tensor([tokens.shape[1]])
        # print(f'prefix ids: {prefix_index}')
        prefix_indices.append(prefix_index.tolist())
        
        # print("eod length",len(eod_index))
        # assert eod_index.shape[0] > 1
        padding_length = tokens.shape[1]-tokens_wo_ans.shape[0]
        padding_seq = torch.tensor(tokenizer.pad).repeat(padding_length).to(tokens.device)
        # tokens[b] = torch.cat((tokens[b, eod_index[0]+1:eod_index[-1]+1], padding_seq)) 
        tokens[b] = torch.cat((tokens_wo_ans, padding_seq))

    # Get the masks and postition ids.
    attention_mask, loss_mask, position_ids = get_sft_ltor_masks_and_position_ids(
        tokens,
        tokenizer.eod,
        tokenizer.pad,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss,
        prefix_indices=None,
        loss_on_targets_only=args.loss_on_targets_only
    )
    return tokens, labels, loss_mask, attention_mask, position_ids


def get_ltor_sft_no_split_prompt(
    tokens, tokenizer,
    reset_attention_mask=True, reset_position_ids=True,
    multiturn_split_id=-100
    ):
  
  labels = tokens[:, 1:].contiguous()
  tokens = tokens[:, :-1].contiguous()
  
  micro_batch_size, seq_length = tokens.size()
  # Loss mask 处理, 
  loss_mask = torch.ones(tokens.size(), dtype=torch.float, device=tokens.device)
  
  loss_mask[tokens==tokenizer.eod] = 0.0
  
  #  Attention mask.
  attention_mask = torch.tril(
    torch.ones((micro_batch_size, seq_length, seq_length),
                device=tokens.device)).view(micro_batch_size, 1, seq_length,
                                          seq_length)
                
  # Position ids.
  position_ids = torch.arange(seq_length, dtype=torch.long, device=tokens.device)
  position_ids = position_ids.unsqueeze(0).expand_as(tokens)
  
  # Prevent cross document interactions
  for b in range(micro_batch_size):
    eod_index = (tokens[b] == tokenizer.eod).nonzero().squeeze(1).tolist()
    if reset_attention_mask:
      for i in eod_index:
        attention_mask[b, 0, (i + 1):, :(i + 1)] = 0

  # 去掉ans token
    if reset_position_ids:
      prev_index = 0
      for i in eod_index:
        position_ids[b, (i + 1):] -= (i + 1 - prev_index)
        prev_index = i + 1
    
  attention_mask = (attention_mask < 0.5).type(torch.uint8)

  tokens[tokens==multiturn_split_id] = tokenizer.eod
  labels[labels==multiturn_split_id] = tokenizer.eod
  
  return (tokens, position_ids, attention_mask), (labels, loss_mask)



# loss mask must remove ans and prompt
def get_ltor_sft_with_split_prompt(
    tokens, tokenizer,reset_attention_mask=True, reset_position_ids=True,
    prompt_split_id=-200,
    multiturn_split_id=-100,
    ):
  """
  需要将prompt的loss进行mask, 这里-200表示prompt位置, -100表示每一轮对话的结束, eod表示整个对话的结束。
  """
  micro_batch_size, seq_length = tokens.size()
  # Loss mask.
  loss_mask = torch.ones(tokens.size(), dtype=torch.float, device=tokens.device)

  # loss_mask_list = []
  # attention_mask_list = []
  
  def pad(cur_tokens, max_len, padding_value):
    assert len(cur_tokens.shape) == 1
    padding_len = max_len - cur_tokens.shape[0]
    if padding_len > 0:
      output_tokens = torch.cat(
        (cur_tokens, 
         torch.tensor(
           [padding_value]*padding_len, device=cur_tokens.device
            ))
        )
      return output_tokens
    return cur_tokens
  
  for b in range(micro_batch_size):
    # Find indices where eod and ans is.
    eod_index = (tokens[b] == tokenizer.eod).nonzero().squeeze(1)
    ans_index = (tokens[b] == prompt_split_id).nonzero().squeeze(1)
    eod_index_turn = (tokens[b] == multiturn_split_id).nonzero().squeeze(1)
    eod_index = torch.cat((eod_index,eod_index_turn))  # eod 部分由eod与-100组成

    # Prompt loss to 0.
    if not (tensor_empty_check(eod_index) or tensor_empty_check(ans_index)):
      eod_index_list = [{"index":i, "type":"eod"} for i in eod_index.tolist()]
      ans_index_list = [{"index":i, "type":"ans"} for i in ans_index.tolist()]
      eod_ans_index = eod_index_list + ans_index_list
      # min_len = min(len(eod_index, ans_index))
      eod_ans_index.sort(key=lambda x: x["index"])
      eod_ans_list = []
      prev_type = "eod"
      prev_index = 0
      for eod_or_ans in eod_ans_index:
        if eod_or_ans["type"] == "ans" and prev_type == "eod":
            eod_ans_list.append((prev_index, eod_or_ans["index"]))
        prev_type = eod_or_ans["type"]
        prev_index = eod_or_ans["index"]
      if prev_type == "eod":
        eod_ans_list.append((prev_index, loss_mask.shape[1]))
      
      for eod_ans_pair in eod_ans_list:
        # 不屏蔽eod loss，来预测下一个token
        loss_mask[b, eod_ans_pair[0]+1:eod_ans_pair[1]] = 0.0
    
    tokens_wo_ans_index = (tokens[b] != prompt_split_id)
    tokens[b] = pad(
      tokens[b][tokens_wo_ans_index], tokens.shape[1], tokenizer.pad)
    loss_mask[b] = pad(
      loss_mask[b][tokens_wo_ans_index], tokens.shape[1], 0.0)
  

  tokens, labels = tokens[:, :-1].contiguous(),tokens[:, 1:].contiguous()
  # loss mask要和label对齐
  loss_mask = loss_mask[:, 1:]

  
  # Reset tokens, micro batch size and sequence length.
  micro_batch_size, seq_length = tokens.size()
  
  # Attention mask.
  attention_mask = torch.tril(
    torch.ones((micro_batch_size, seq_length, seq_length),
                device=tokens.device)).view(micro_batch_size, 1, seq_length,
                                          seq_length)
                
  # Position ids.
  position_ids = torch.arange(seq_length, dtype=torch.long, device=tokens.device)
  position_ids = position_ids.unsqueeze(0).expand_as(tokens)
  
  # Prevent cross document interactions.
  for b in range(micro_batch_size):
    eod_index = (tokens[b] == tokenizer.eod).nonzero().squeeze(1).tolist()
    
    if reset_attention_mask:
      for i in eod_index:
        attention_mask[b, 0, (i + 1):, :(i + 1)] = 0

    # reset position ids
    if reset_position_ids:
      prev_index = 0
      for i in eod_index:
        position_ids[b, (i + 1):] -= (i + 1 - prev_index)
        prev_index = i + 1
    
  attention_mask = (attention_mask < 0.5).type(torch.uint8)

  # multiturn_split_id标签变成eod
  tokens[tokens==multiturn_split_id] = tokenizer.eod
  labels[labels==multiturn_split_id] = tokenizer.eod


  return (tokens, position_ids, attention_mask), (labels, loss_mask)


  
def get_batch_pipe_train(data):
    """Modification of `get_batch` to work on `next(data_iterator)` instead of `data_iterator`"""
    args = get_args()
    tokenizer = get_tokenizer()

    # Items and their type.
    keys = ['text']
    datatype = torch.int64
     
    """
    input
    stage1: 1,2
      
    stage2: 3,4

    stage3: 5,6

    stage4: 7,8
    """
    # Broadcast data.
    data_b = mpu.broadcast_data(keys, data, datatype)

    # Unpack.
    tokens = data_b['text'].long()

    (tokens, position_ids, attention_mask), (labels, loss_mask) = get_ltor_sft_with_split_prompt(
        tokens,
        tokenizer,
        reset_attention_mask=True,
        reset_position_ids=True,
        prompt_split_id=-200,
        multiturn_split_id=-100
    )

    tokens = tokens.contiguous()
    position_ids = position_ids.contiguous()
    attention_mask = attention_mask.contiguous()
    labels = labels.contiguous()
    loss_mask = loss_mask.contiguous()

    return (tokens, position_ids, attention_mask), (labels, loss_mask)

def get_batch_pipe_eval(data):
    """Modification of `get_batch` to work on `next(data_iterator)` instead of `data_iterator`"""
    # 验证阶段的数据构造

    args = get_args()
    tokenizer = get_tokenizer()

    if data is None:
       data = [None]
    else:
       data = [data]

    torch.distributed.broadcast_object_list(
      data,
      mpu.get_tensor_model_parallel_src_rank(),
      group=mpu.get_tensor_model_parallel_group()
    )

    data = data[0]

    prompts = [datum['prompt_ids'] for datum in data]
    answers = [datum['chosen_ids'] for datum in data]

    input_ids_list = []
    loss_mask_list = []
    for prompt, answer in zip(prompts,answers):
      answers += [tokenizer.eod]
      input_ids_list.append(torch.LongTensor(prompt+answer))
      loss_mask_list.append(torch.LongTensor([0]*len(prompt) + [1]*len(answer)))
    
    # 进行拼接
    input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=tokenizer.pad)
    loss_mask = pad_sequence(loss_mask_list, batch_first=True, padding_value=0)

    # pad 到 最大长度, 尽管这样不是高效的
    seq_length = args.seq_length + 1
    if input_ids.shape[1] < seq_length:
       bs,seq_len = input_ids.shape
       input_ids = torch.cat((input_ids,torch.ones((bs,seq_length-seq_len),device=input_ids.device)*tokenizer.pad),dim=-1).long()
       loss_mask = torch.cat((loss_mask,torch.zeros((bs,seq_length-seq_len),device=input_ids.device)),dim=-1).long()
    
    input_ids = input_ids[:,:args.seq_length].cuda()
    loss_mask = loss_mask[:,:args.seq_length].cuda()

    # 确定position_ids 与 attention_mask

    batch_size,maxlen = input_ids.shape
    attention_mask = torch.tril(
        torch.ones((batch_size, maxlen, maxlen),
                  device=input_ids.device)).view(batch_size, 1, maxlen,
                                            maxlen)
    position_ids = torch.arange(maxlen, dtype=torch.long, device=input_ids.device)
    position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
    attention_mask = (attention_mask < 0.5).type(torch.uint8)

    # 张量连续化
    input_ids = input_ids.contiguous()
    position_ids = position_ids.contiguous()
    attention_mask = attention_mask.contiguous()
    labels = labels.contiguous()
    loss_mask = loss_mask.contiguous()

    # print_with_rank('input',input_ids[:5,:5],input_ids.shape)

    # 返回的格式必须是包含两个元素的tuple/list
    return (input_ids, position_ids, attention_mask), (labels, loss_mask)


def get_batch_pipe(data):
  args = get_args()
  if args.is_training:
     return get_batch_pipe_train(data)
  else:
    return get_batch_pipe_train(data)
  

def loss_func(loss_mask, output_tensor):
    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

    # Reduce loss for logging.
    averaged_loss = average_losses_across_data_parallel_group([loss])

    return loss, {'lm loss': averaged_loss[0]}


def forward_step(data_iterator, model):
    """Forward step."""
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch-generator').start()
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(
        data_iterator)
    # print("tokens",tokens, "the shape of tokens", tokens.shape)
    timers('batch-generator').stop()

    output_tensor = model(tokens, position_ids, attention_mask,
                          labels=labels)
    if args.curriculum_learning and args.curriculum_seqlen < args.seq_length:
        loss_mask = loss_mask[:, :args.curriculum_seqlen].contiguous()

    return output_tensor, partial(loss_func, loss_mask)


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()
    train_ds, valid_ds, test_ds = None, None, None

    print_rank_0('> building train, validation, and test datasets for GPT ...')
    # Option 1 of data loading using --data-path
    assert args.data_path, args.data_path

    def load_feats(path):
      with open(path,'rb') as f:
        data = pickle.load(f)
        # print(len(data))
        # print(data[0])
        yield from iter(data)


    if args.data_path:
        train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
            data_prefix=args.data_path,
            data_impl=args.data_impl,
            splits_string=args.split,
            train_valid_test_num_samples=train_val_test_num_samples,
            seq_length=args.seq_length,
            seed=args.seed,
            skip_warmup=(not args.mmap_warmup))

        # 修改验证集与测试集
        if args.valid_data_path:
           valid_ds = Dataset.from_generator(partial(load_feats,args.args.valid_data_path))
           test_ds = valid_ds
           print_rank_0('   new validation sample num: {}'.format(len(valid_ds)))
           print_rank_0('    new test smaple num:       {}'.format(len(test_ds)))
      
    # Option 2 of data loading using --(train|valid|test)-weighted-split-paths
    elif args.train_weighted_split_paths:
        assigned_train_valid_test = []
        if args.train_weighted_split_paths is not None:
            train_ds = []
            assigned_train_valid_test.append("train")
        if args.valid_weighted_split_paths is not None:
            valid_ds = []
            assigned_train_valid_test.append("valid")
        if args.test_weighted_split_paths is not None:
            test_ds = []
            assigned_train_valid_test.append("test")

        for s in assigned_train_valid_test:
            data_groups = zip(eval(f"args.{s}_weighted_split_paths"),
                                eval(f"args.{s}_weighted_split_weights"),
                                eval(f"args.{s}_weighted_split_splits"),
                                eval(f"args.{s}_weighted_split_names"))
            for paths, weights, splits, name in data_groups:
                d = build_dataset_group(name, paths, weights, splits,
                                        args.data_impl,
                                        train_val_test_num_samples,
                                        args.seq_length, args.seed,
                                        (not args.mmap_warmup),
                                        train_valid_test=s)
                eval(f"{s}_ds").append(d)
    else:
        raise NotImplementedError("No dataloading argument passed")

    print_rank_0("> finished creating GPT datasets ...")
    return train_ds, valid_ds, test_ds



def sft_get_params_for_weight_decay_optimization(modules):
  """Divide params into with-weight-decay and without-weight-decay groups.
    Layernorms and baises will have no weight decay but the rest will.
    """

  weight_decay_params = {'params': []}
  no_weight_decay_params = {'params': [], 'weight_decay': 0.0}
  for module in modules:
    for module_ in module.modules():
      if isinstance(module_, LayerNorm):
        no_weight_decay_params['params'].extend(
            [p for p in list(module_._parameters.values()) if p is not None])
      else:
        weight_decay_params['params'].extend([
            p for n, p in list(module_._parameters.items())
            if p is not None and n != 'bias'
        ])
        no_weight_decay_params['params'].extend([
            p for n, p in list(module_._parameters.items())
            if p is not None and n == 'bias'
        ])
  
  # XXX: temp hack to workaround the crash in apex FusedAdam's multi_tensor_applier
  #
  # it crashes when the param count is larger than a certain size which we hit at 200B over 80
  # A100 gpus - I think around 2.7B per gpu, so halving it works around the issue
  # param_count = len(weight_decay_params['params'])
  # first_half = weight_decay_params['params'][:param_count // 2]
  # second_half = weight_decay_params['params'][param_count // 2:]

  # first_half = {'params': first_half}
  # second_half = {'params': second_half}
  # if mpu.is_pipeline_last_stage():
  #   # for module in modules:
  #   #   for n,v in module.named_parameters():
  #   #     print(n,v.shape)
  #   # print(
  #   #   f"first_half num: {len(first_half['params'])}, second_half: {second_half['params']},no_weight_decay_params: {no_weight_decay_params}")
  
  #   print(sg)
  print_with_rank(f'weight_decay_params: {len(weight_decay_params["params"])},  no_weight_decay_params: {len(no_weight_decay_params["params"])}')
  # torch.distributed.barrier()
  # print(srg)
  return weight_decay_params, no_weight_decay_params


def lora_param_groups_provider(modules):
  args = get_args()
  if args.use_lora:
    lora_params = []
    for module in modules:
        for n,v in module.named_parameters():
          if 'lora_A' in n or 'lora_B' in n:
            lora_params.append(v)
          else:
            v.requires_grad = False
    print('lora_params num',len(lora_params))
    return [{'params':lora_params}]
  else:
    return sft_get_params_for_weight_decay_optimization(modules)

@record
def main():
  pretrain(
    train_valid_test_datasets_provider, 
    model_provider,
    forward_step,
    param_groups_provider=lora_param_groups_provider,
    extra_args_provider=add_step1_sft_args
  )

if __name__ == "__main__":
    main()

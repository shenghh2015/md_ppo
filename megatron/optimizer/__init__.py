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

from apex.optimizers import FusedAdam as Adam
from apex.optimizers import FusedSGD as SGD
from deepspeed.ops.adam import DeepSpeedCPUAdam
from torch.optim import Adam as torch_adam
from functools import partial

from megatron import get_args
from megatron.model.fused_layer_norm import MixedFusedLayerNorm as LayerNorm

from .grad_scaler import ConstantGradScaler, DynamicGradScaler
from .optimizer import Float16OptimizerWithFloat16Params, FP32Optimizer, CPUAdam
from megatron.utils import print_rank_0

from megatron import mpu,print_with_rank
import torch


def _get_params_for_weight_decay_optimization(modules):
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
  param_count = len(weight_decay_params['params'])
  first_half = weight_decay_params['params'][:param_count // 2]
  second_half = weight_decay_params['params'][param_count // 2:]

  first_half = {'params': first_half}
  second_half = {'params': second_half}
  # if mpu.is_pipeline_last_stage():
  #   # for module in modules:
  #   #   for n,v in module.named_parameters():
  #   #     print(n,v.shape)
  #   # print(
  #   #   f"first_half num: {len(first_half['params'])}, second_half: {second_half['params']},no_weight_decay_params: {no_weight_decay_params}")
  
  #   print(sg)
  # print_with_rank(f'first_half: {len(first_half["params"])}, second_half: {len(second_half["params"])}, no_weight_decay_params: {len(no_weight_decay_params["params"])}')
  # torch.distributed.barrier()
  # print(srg)
  return first_half, second_half, no_weight_decay_params

  #return weight_decay_params, no_weight_decay_params


def get_megatron_optimizer(model,param_groups_provider=None):
  args = get_args()

  # if args.cpu_optimizer:
  #     raise NotImplementedError('need to add cpu adam')

  # Base optimizer.
  from megatron import print_with_rank
  if param_groups_provider is None:
    param_groups = _get_params_for_weight_decay_optimization(model)
  else:
    param_groups = param_groups_provider(model)

  # for n,v in model[0].named_parameters():
  #   print_with_rank(n,v.shape)
  
  # for group in param_groups:
  #   for p in group['params']:
  #     print(p.name,p.shape)
  # print(sfg)
  
  # print_rank_0(f'param_groups:{param_groups}')
  # print(sdg)
  # print_with_rank('optimi group: ',len(param_groups))
  # import torch
  # torch.distributed.barrier()
  # print(sfb)
  if args.optimizer == 'adam':
    if args.use_bnb_optimizer:
      import bitsandbytes as bnb
      adam_optimizer = partial(bnb.optim.Adam8bit,min_8bit_size=args.min_8bit_size)
    else:
      if args.cpu_optimizer:
        if args.deepspeed:
          adam_optimizer = CPUAdam
          import json,io
          with io.open(args.deepspeed_config, "r", encoding="utf-8") as f:
            ds_config = json.load(f)
          
          # zero stage >0 的时候使用deepspeed的cpu 优化器
          if ds_config['zero_optimization']['stage'] > 0:
            adam_optimizer = DeepSpeedCPUAdam
        
          # return None
          # from functools import partial
          # adam_optimizer =DeepSpeedCPUAdam # partial(DeepSpeedCPUAdam,fp32_optimizer_states=False)
        else:
          adam_optimizer = CPUAdam
      else:
        
        adam_optimizer = Adam

    
    # import psutil,time
    # import torch.distributed as dist
    # print(f'rank: {dist.get_rank()}开始实例化优化器,当前cpu内存: ', psutil.Process().memory_full_info().rss/1024/1024/1024)
    # time.sleep(20)
    # import inspect
    # print('adam_optimizer sfgs',type(adam_optimizer),inspect.getabsfile(adam_optimizer))
    optimizer = adam_optimizer(param_groups,
                               lr=args.lr,
                               weight_decay=args.weight_decay,
                               betas=(args.adam_beta1, args.adam_beta2),
                               eps=args.adam_eps)
    # print_with_rank('after optimi group: ',len(param_groups))
    # import torch
    # torch.distributed.barrier()
    # print(sg)
    # print(f'rank: {dist.get_rank()}完成实例化优化器,当前cpu内存: ', psutil.Process().memory_full_info().rss/1024/1024/1024)
    # time.sleep(20)
  elif args.optimizer == 'sgd':
    optimizer = SGD(param_groups,
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                    momentum=args.sgd_momentum)
  else:
    raise Exception('{} optimizer is not supported.'.format(args.optimizer))
  
  # import inspect
  # print(f'use optimizer class: {optimizer}',optimizer.__module__)
  
  if args.deepspeed:
    return optimizer

  # Determine whether the params have main-grad field.
  params_have_main_grad = False
  if args.DDP_impl == 'local':
    params_have_main_grad = True

  if args.fp16 or args.bf16:

    # Grad scaler:
    #    if loss-scale is provided, instantiate the constant scaler.
    #    if we are using fp16 and loss-scale is not present, use a
    #       dynamic scaler.
    #    otherwise we are running in bf16 with no loss-scale so
    #       leave it as None.
    grad_scaler = None
    # Constant loss scale.
    if args.loss_scale:
      grad_scaler = ConstantGradScaler(args.loss_scale)
    # Dynamic loss scale.
    else:
      if args.fp16:
        grad_scaler = DynamicGradScaler(initial_scale=args.initial_loss_scale,
                                        min_scale=args.min_loss_scale,
                                        growth_factor=2.0,
                                        backoff_factor=0.5,
                                        growth_interval=args.loss_scale_window,
                                        hysteresis=args.hysteresis)

    # Megatron optimizer.
    return Float16OptimizerWithFloat16Params(optimizer, args.clip_grad,
                                             args.log_num_zeros_in_grad,
                                             params_have_main_grad, args.bf16,
                                             grad_scaler)

  # FP32.
  # for param_group in optimizer.param_groups:
  #   print('param group before fp32',param_group['lr'], len(param_group['params']))
  optimizer = FP32Optimizer(optimizer, args.clip_grad, args.log_num_zeros_in_grad,
                       params_have_main_grad)
  # for param_group in optimizer.param_groups:
  #   print('param group after fp32',param_group['lr'], len(param_group['params']))
  return optimizer

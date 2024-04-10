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

from commons import print_separator
from commons import initialize_distributed
import megatron.mpu as mpu
import torch
import sys
from megatron import print_with_rank

def test_initialize_model_parallel(tensor_model_parallel_size):

  if torch.distributed.get_rank() == 0:
    print('> testing initialize_model_parallel with size {} ...'.format(
        tensor_model_parallel_size))
  tensor_model_parallel_size_ = min(tensor_model_parallel_size,
                                    torch.distributed.get_world_size())
  assert not mpu.model_parallel_is_initialized()
  mpu.initialize_model_parallel(tensor_model_parallel_size_)
  assert mpu.model_parallel_is_initialized()

  # Checks.
  def check(group, world_size, rank):
    assert world_size == torch.distributed.get_world_size(group=group)
    assert rank == torch.distributed.get_rank(group=group)

  # Model parallel.
  world_size = tensor_model_parallel_size_
  rank = torch.distributed.get_rank() % tensor_model_parallel_size_
  assert world_size == mpu.get_tensor_model_parallel_world_size()
  assert rank == mpu.get_tensor_model_parallel_rank()
  check(mpu.get_tensor_model_parallel_group(), world_size, rank)

  # Data parallel.
  world_size = torch.distributed.get_world_size(
  ) // tensor_model_parallel_size_
  rank = torch.distributed.get_rank() // tensor_model_parallel_size
  assert world_size == mpu.get_data_parallel_world_size()
  assert rank == mpu.get_data_parallel_rank()
  check(mpu.get_data_parallel_group(), world_size, rank)

  # Reset groups
  mpu.destroy_model_parallel()

  torch.distributed.barrier()
  if torch.distributed.get_rank() == 0:
    print('>> passed the test :-)')


def test_get_tensor_model_parallel_src_rank(tensor_model_parallel_size_):

  if torch.distributed.get_rank() == 0:
    print(
        '> testing get_tensor_model_parallel_src_rank with size {} ...'.format(
            tensor_model_parallel_size_))
  tensor_model_parallel_size = min(tensor_model_parallel_size_,
                                   torch.distributed.get_world_size())
  assert not mpu.model_parallel_is_initialized()
  mpu.initialize_model_parallel(tensor_model_parallel_size)
  assert mpu.model_parallel_is_initialized()

  # Checks
  src_rank = torch.distributed.get_rank() - mpu.get_tensor_model_parallel_rank(
  )
  assert mpu.get_tensor_model_parallel_src_rank() == src_rank

  # Reset groups
  mpu.destroy_model_parallel()

  torch.distributed.barrier()
  if torch.distributed.get_rank() == 0:
    print('>> passed the test :-)')

def run_test(
        tensor_model_parallel_size: int,
        pipeline_model_parallel_size:int):
    print_separator(f'> Test: TP={tensor_model_parallel_size}, PP={pipeline_model_parallel_size}')
    mpu.initialize_model_parallel(
            tensor_model_parallel_size,
            pipeline_model_parallel_size) # 并行初始化
    world_size = torch.distributed.get_world_size() # world_size, 总GPU数量
    global_rank = torch.distributed.get_rank()      # 当前GPU的编号
    tp_world_size = mpu.get_tensor_model_parallel_world_size() # 每个张量并行组中GPU的数量
    pp_world_size = mpu.get_pipeline_model_parallel_world_size() # 每个流水线并行组中GPU的数量
    dp_world_size = mpu.get_data_parallel_world_size() # 每个数据并行组中的GPU数量
    tp_rank = mpu.get_tensor_model_parallel_rank()   # 在张量并行组中的编号
    pp_rank = mpu.get_pipeline_model_parallel_rank() # 在流水线并行组中的编号
    dp_rank = mpu.get_data_parallel_rank()           # 在数据并行组中的编号
    tp_group = mpu.get_tensor_model_parallel_group() 
    tp_group = torch.distributed.distributed_c10d._pg_group_ranks[tp_group] # 当前GPU所在张量并行组的映射字典
    pp_group = mpu.get_pipeline_model_parallel_group()
    pp_group = torch.distributed.distributed_c10d._pg_group_ranks[pp_group] # 当前GPU所在流水线并行组的映射字典
    dp_group = mpu.get_data_parallel_group()
    dp_group = torch.distributed.distributed_c10d._pg_group_ranks[dp_group] # 当前GPU所在数据并行组的映射字典
    torch.distributed.barrier()
    info = f"="*20 + \
            f"\n> global_rank={global_rank}\n" + \
            f"> world_size={world_size}\n" + \
            f"> tp_world_size={tp_world_size}\n" + \
            f"> pp_world_size={pp_world_size}\n" + \
            f"> dp_world_size={dp_world_size}\n" + \
            f"> tp_rank={tp_rank}\n" + \
            f"> pp_rank={pp_rank}\n" + \
            f"> dp_rank={dp_rank}\n" + \
            f"> tp_group={tp_group}\n" + \
            f"> pp_group={pp_group}\n" + \
            f"> dp_group={dp_group}\n"
    print(info, flush=True)
    torch.distributed.barrier()

if __name__ == '__main__':

  initialize_distributed()
  tensor_model_parallel_size = 2
  pipeline_model_parallel_size = 4
  run_test(tensor_model_parallel_size, pipeline_model_parallel_size)
  # world_size = torch.distributed.get_world_size()
  # tensor_model_parallel_size = 1
  # while tensor_model_parallel_size <= world_size:
  #   print_separator('test initialize model parallel')
  #   test_initialize_model_parallel(tensor_model_parallel_size)
  #   print_separator('test model parallel source rank')
  #   test_get_tensor_model_parallel_src_rank(tensor_model_parallel_size)
  #   tensor_model_parallel_size *= 2

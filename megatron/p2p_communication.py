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

from functools import reduce
from typing import Optional, List, Union, Callable, Tuple
import operator
import torch

from megatron import get_args
from megatron import mpu,print_with_rank



Shape = Union[List[int], torch.Size]

def _communicate_shapes(tensor_send_next, tensor_send_prev,
                        recv_prev, recv_next,
                        use_ring_exchange_p2p):
    """Communicate tensor shapes between stages. Used to communicate
    tensor shapes before the actual tensor communication happens.
    This is required when the sequence lengths across micro batches
    are not uniform.

    Takes the following arguments:
        tensor_send_next: tensor to send to next rank (no tensor sent if
                          set to None).
        tensor_send_prev: tensor to send to prev rank (no tensor sent if
                          set to None).
        recv_prev: boolean for whether tensor should be received from
                   previous rank.
        recv_next: boolean for whether tensor should be received from
                   next rank.
    Returns:
        (recv_prev_shape, recv_next_shape)
    """

    recv_prev_shape_tensor = None
    recv_next_shape_tensor = None
    send_prev_shape_tensor = None
    send_next_shape_tensor = None
    if recv_prev:
        recv_prev_shape_tensor = torch.empty((3),
                                             device=torch.cuda.current_device(),
                                             dtype=torch.int64)
    if recv_next:
        recv_next_shape_tensor = torch.empty((3),
                                             device=torch.cuda.current_device(),
                                             dtype=torch.int64)
    if tensor_send_prev is not None:
        send_prev_shape_tensor = torch.tensor(tensor_send_prev.size(),
                                              device=torch.cuda.current_device(),
                                              dtype=torch.int64)
    if tensor_send_next is not None:
        send_next_shape_tensor = torch.tensor(tensor_send_next.size(),
                                              device=torch.cuda.current_device(),
                                              dtype=torch.int64)

    if use_ring_exchange_p2p:
        torch.distributed.ring_exchange(tensor_send_prev=send_prev_shape_tensor,
                                        tensor_recv_prev=recv_prev_shape_tensor,
                                        tensor_send_next=send_next_shape_tensor,
                                        tensor_recv_next=recv_next_shape_tensor,
                                        group=mpu.get_pipeline_model_parallel_group())
    else:
        ops = []
        if send_prev_shape_tensor is not None:
            send_prev_op = torch.distributed.P2POp(
                torch.distributed.isend, send_prev_shape_tensor,
                mpu.get_pipeline_model_parallel_prev_rank())
            ops.append(send_prev_op)
        if recv_prev_shape_tensor is not None:
            recv_prev_op = torch.distributed.P2POp(
                torch.distributed.irecv, recv_prev_shape_tensor,
                mpu.get_pipeline_model_parallel_prev_rank())
            ops.append(recv_prev_op)
        if send_next_shape_tensor is not None:
            send_next_op = torch.distributed.P2POp(
                torch.distributed.isend, send_next_shape_tensor,
                mpu.get_pipeline_model_parallel_next_rank())
            ops.append(send_next_op)
        if recv_next_shape_tensor is not None:
            recv_next_op = torch.distributed.P2POp(
                torch.distributed.irecv, recv_next_shape_tensor,
                mpu.get_pipeline_model_parallel_next_rank())
            ops.append(recv_next_op)
        if len(ops) > 0:
            reqs = torch.distributed.batch_isend_irecv(ops)
            for req in reqs:
                req.wait()

        # To protect against race condition when using batch_isend_irecv().
        # should take this out once the bug with batch_isend_irecv is resolved.
        torch.cuda.synchronize()

    recv_prev_shape = [0, 0, 0]
    if recv_prev_shape_tensor is not None:
        recv_prev_shape = recv_prev_shape_tensor.tolist()

    recv_next_shape = [0, 0, 0]
    if recv_next_shape_tensor is not None:
        recv_next_shape = recv_next_shape_tensor.tolist()

    return recv_prev_shape, recv_next_shape


def _communicate(tensor_send_next,
                 tensor_send_prev,
                 recv_prev,
                 recv_next,
                 use_ring_exchange=False):
  """Communicate tensors between stages. Used as helper method in other
    communication methods that are used in megatron/schedules.py.

    Takes the following arguments:
        tensor_send_next: tensor to send to next rank (no tensor sent if
                          set to None).
        tensor_send_prev: tensor to send to prev rank (no tensor sent if
                          set to None).
        recv_prev: boolean for whether tensor should be received from
                   previous rank.
        recv_next: boolean for whether tensor should be received from
                   next rank.
        use_ring_exchange: boolean for whether torch.distributed.ring_exchange()
                           API should be used.

    Returns:
        (tensor_recv_prev, tensor_recv_next)
    """
  args = get_args()
  # Create placeholder tensors for receive in forward and backward directions
  # if needed.
  tensor_recv_prev = None
  tensor_recv_next = None

  # 对于可变的张量维数，需要先去获取一下变量维数信息
  if args.variable_seq_lengths:
    recv_prev_shape, recv_next_shape = \
            _communicate_shapes(tensor_send_next,
                                tensor_send_prev,
                                recv_prev,
                                recv_next,
                                use_ring_exchange
                                )
  else:
    recv_prev_shape,recv_next_shape =  (args.seq_length, args.micro_batch_size, args.hidden_size),\
      (args.seq_length, args.micro_batch_size, args.hidden_size)
    
  # tensor_shape = (args.seq_length, args.micro_batch_size, args.hidden_size)
  # add by zx
  # tensor_shape = (1, args.micro_batch_size, args.hidden_size)

  def _get_tensor_chunk_shape(tensor_shape):
    if args.scatter_gather_tensors_in_pipeline:
      
      tensor_chunk_shape = reduce(operator.mul, tensor_shape, 1) // \
          mpu.get_tensor_model_parallel_world_size()
      # print('scatter_gather_tensors_in_pipeline',tensor_chunk_shape)
    else:
      tensor_chunk_shape = tensor_shape
    return tensor_chunk_shape

  dtype = args.params_dtype
  if args.fp32_residual_connection:
    dtype = torch.float
  if recv_prev:
    # print_with_rank('recv_prev_shape',recv_prev_shape)
    tensor_recv_prev = torch.empty(_get_tensor_chunk_shape(recv_prev_shape),
                                   requires_grad=True,
                                   device=torch.cuda.current_device(),
                                   dtype=dtype)
  if recv_next:
    tensor_recv_next = torch.empty(_get_tensor_chunk_shape(recv_next_shape),
                                   requires_grad=True,
                                   device=torch.cuda.current_device(),
                                   dtype=dtype)

  # Split tensor into smaller chunks if using scatter-gather optimization.
  if args.scatter_gather_tensors_in_pipeline:
    if tensor_send_next is not None:
      # print_with_rank('tensor_send_next shape',tensor_send_next.shape)
      tensor_send_next = mpu.split_tensor_into_1d_equal_chunks(
          tensor_send_next)

    if tensor_send_prev is not None:
      tensor_send_prev = mpu.split_tensor_into_1d_equal_chunks(
          tensor_send_prev)

  # Send tensors in both the forward and backward directions as appropriate.
  if use_ring_exchange:
    torch.distributed.ring_exchange(
        tensor_send_prev=tensor_send_prev,
        tensor_recv_prev=tensor_recv_prev,
        tensor_send_next=tensor_send_next,
        tensor_recv_next=tensor_recv_next,
        group=mpu.get_pipeline_model_parallel_group())
  else:
    ops = []
    if tensor_send_prev is not None:
      send_prev_op = torch.distributed.P2POp(
          torch.distributed.isend, tensor_send_prev,
          mpu.get_pipeline_model_parallel_prev_rank())
      ops.append(send_prev_op)
    if tensor_recv_prev is not None:
      # print_with_rank('tensor_recv_prev.shape',tensor_recv_prev.shape)
      recv_prev_op = torch.distributed.P2POp(
          torch.distributed.irecv, tensor_recv_prev,
          mpu.get_pipeline_model_parallel_prev_rank())
      ops.append(recv_prev_op)
    if tensor_send_next is not None:
      send_next_op = torch.distributed.P2POp(
          torch.distributed.isend, tensor_send_next,
          mpu.get_pipeline_model_parallel_next_rank())
      ops.append(send_next_op)
    if tensor_recv_next is not None:
      recv_next_op = torch.distributed.P2POp(
          torch.distributed.irecv, tensor_recv_next,
          mpu.get_pipeline_model_parallel_next_rank())
      ops.append(recv_next_op)
    if len(ops) > 0:
      # print_with_rank('ops',ops)
      reqs = torch.distributed.batch_isend_irecv(ops)
      for req in reqs:
        req.wait()
  # To protect against race condition when using batch_isend_irecv().
  torch.cuda.synchronize()

  # add by zx
  # if tensor_recv_prev is not None:
  #   tensor_shape =  tensor_recv_prev.shape
  #   print('tensor_shape after',tensor_shape)

  # if tensor_recv_next is not None:
  #   tensor_shape = tensor_recv_next.shape
  #   print('tensor_shape after',tensor_shape)

  # If using scatter-gather optimization, gather smaller chunks.
  if args.scatter_gather_tensors_in_pipeline:
    if recv_prev:
      tensor_recv_prev = mpu.gather_split_1d_tensor(tensor_recv_prev).view(
          recv_prev_shape).requires_grad_()

    if recv_next:
      tensor_recv_next = mpu.gather_split_1d_tensor(tensor_recv_next).view(
          recv_next_shape).requires_grad_()

  return tensor_recv_prev, tensor_recv_next


def recv_forward(timers=None):
  """Receive tensor from previous rank in pipeline (forward receive)."""
  if mpu.is_pipeline_first_stage():
    input_tensor = None
  else:
    if timers is not None:
      timers('forward-recv').start()
    input_tensor, _ = _communicate(tensor_send_next=None,
                                   tensor_send_prev=None,
                                   recv_prev=True,
                                   recv_next=False)
    if timers is not None:
      timers('forward-recv').stop()
  return input_tensor


def recv_backward(timers=None):
  """Receive tensor from next rank in pipeline (backward receive)."""
  if mpu.is_pipeline_last_stage():
    output_tensor_grad = None
  else:
    if timers is not None:
      timers('backward-recv').start()
    _, output_tensor_grad = _communicate(tensor_send_next=None,
                                         tensor_send_prev=None,
                                         recv_prev=False,
                                         recv_next=True)
    if timers is not None:
      timers('backward-recv').stop()
  return output_tensor_grad


def send_forward(output_tensor, timers=None):
  """Send tensor to next rank in pipeline (forward send)."""
  if not mpu.is_pipeline_last_stage():
    if timers is not None:
      timers('forward-send').start()
    _communicate(tensor_send_next=output_tensor,
                 tensor_send_prev=None,
                 recv_prev=False,
                 recv_next=False)
    if timers is not None:
      timers('forward-send').stop()


def send_backward(input_tensor_grad, timers=None):
  """Send tensor to previous rank in pipeline (backward send)."""
  if not mpu.is_pipeline_first_stage():
    if timers is not None:
      timers('backward-send').start()
    _communicate(tensor_send_next=None,
                 tensor_send_prev=input_tensor_grad,
                 recv_prev=False,
                 recv_next=False)
    if timers is not None:
      timers('backward-send').stop()


def send_forward_recv_backward(output_tensor, timers=None):
  """Batched send and recv with next rank in pipeline."""
  if mpu.is_pipeline_last_stage():
    output_tensor_grad = None
  else:
    if timers is not None:
      timers('forward-send-backward-recv').start()
    _, output_tensor_grad = _communicate(tensor_send_next=output_tensor,
                                         tensor_send_prev=None,
                                         recv_prev=False,
                                         recv_next=True)
    if timers is not None:
      timers('forward-send-backward-recv').stop()
  return output_tensor_grad


def send_backward_recv_forward(input_tensor_grad, timers=None):
  """Batched send and recv with previous rank in pipeline."""
  if mpu.is_pipeline_first_stage():
    input_tensor = None
  else:
    if timers is not None:
      timers('backward-send-forward-recv').start()
    input_tensor, _ = _communicate(tensor_send_next=None,
                                   tensor_send_prev=input_tensor_grad,
                                   recv_prev=True,
                                   recv_next=False)
    if timers is not None:
      timers('backward-send-forward-recv').stop()
  return input_tensor


def send_forward_recv_forward(output_tensor, recv_prev, timers=None):
  """Batched recv from previous rank and send to next rank in pipeline."""
  if timers is not None:
    timers('forward-send-forward-recv').start()
  input_tensor, _ = _communicate(tensor_send_next=output_tensor,
                                 tensor_send_prev=None,
                                 recv_prev=recv_prev,
                                 recv_next=False)
  if timers is not None:
    timers('forward-send-forward-recv').stop()
  return input_tensor


def send_backward_recv_backward(input_tensor_grad, recv_next, timers=None):
  """Batched recv from next rank and send to previous rank in pipeline."""
  if timers is not None:
    timers('backward-send-backward-recv').start()
  _, output_tensor_grad = _communicate(tensor_send_next=None,
                                       tensor_send_prev=input_tensor_grad,
                                       recv_prev=False,
                                       recv_next=recv_next)
  if timers is not None:
    timers('backward-send-backward-recv').stop()
  return output_tensor_grad


def send_forward_backward_recv_forward_backward(output_tensor,
                                                input_tensor_grad,
                                                recv_prev,
                                                recv_next,
                                                timers=None):
  """Batched send and recv with previous and next ranks in pipeline."""
  if timers is not None:
    timers('forward-backward-send-forward-backward-recv').start()
  input_tensor, output_tensor_grad = _communicate(
      tensor_send_next=output_tensor,
      tensor_send_prev=input_tensor_grad,
      recv_prev=recv_prev,
      recv_next=recv_next)
  if timers is not None:
    timers('forward-backward-send-forward-backward-recv').stop()
  return input_tensor, output_tensor_grad

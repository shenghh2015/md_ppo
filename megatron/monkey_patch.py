"""
一些补丁程序
"""
from typing import Union, Tuple, Any, Callable, Iterator, Set, Optional, overload, TypeVar, Mapping, Dict, List
from collections import OrderedDict
from torch.nn.modules.module import _IncompatibleKeys
import torch
import warnings

def load_state_dir(self, load_dir, strict=True):
  # deepspeed.pipe.PipelineModule 的补丁,主要修复
  for idx, layer in enumerate(self.forward_funcs):
      # Functions, etc. will not have state_dicts
      if not hasattr(layer, 'load_state_dict'):
          continue

      # get all checkpoint files for the layer.
      model_ckpt_list = self.ckpt_layer_path_list(load_dir, idx)
      mp_rank = self._grid.get_slice_parallel_rank()
      mp_world_size = self._grid.get_slice_parallel_world_size()

      from deepspeed.runtime.state_dict_factory import SDLoaderFactory
      sd_loader = SDLoaderFactory.get_sd_loader(model_ckpt_list, version=2.0)
      load_path, checkpoint, _ = sd_loader.load(mp_world_size, mp_rank, module_key=None, is_pipe_parallel=True)

      layer.load_state_dict(checkpoint,strict=strict)

      # if self._grid.data_parallel_id == 0:
      #     logger.info(
      #         f'RANK={self.global_rank} Loaded layer={idx+self._local_start} file={load_path}'
      #     )
  self._synchronize_tied_weights()


def load_state_dict(self, state_dict: Mapping[str, Any],
                        strict: bool = True):
  r"""Copies parameters and buffers from :attr:`state_dict` into
  this module and its descendants. If :attr:`strict` is ``True``, then
  the keys of :attr:`state_dict` must exactly match the keys returned
  by this module's :meth:`~torch.nn.Module.state_dict` function.

  Args:
      state_dict (dict): a dict containing parameters and
          persistent buffers.
      strict (bool, optional): whether to strictly enforce that the keys
          in :attr:`state_dict` match the keys returned by this module's
          :meth:`~torch.nn.Module.state_dict` function. Default: ``True``

  Returns:
      ``NamedTuple`` with ``missing_keys`` and ``unexpected_keys`` fields:
          * **missing_keys** is a list of str containing the missing keys
          * **unexpected_keys** is a list of str containing the unexpected keys

  Note:
      If a parameter or buffer is registered as ``None`` and its corresponding key
      exists in :attr:`state_dict`, :meth:`load_state_dict` will raise a
      ``RuntimeError``.
  """
  if not isinstance(state_dict, Mapping):
      raise TypeError("Expected state_dict to be dict-like, got {}.".format(type(state_dict)))

  missing_keys: List[str] = []
  unexpected_keys: List[str] = []
  error_msgs: List[str] = []

  # copy state_dict so _load_from_state_dict can modify it
  metadata = getattr(state_dict, '_metadata', None)
  state_dict = OrderedDict(state_dict)
  if metadata is not None:
      # mypy isn't aware that "_metadata" exists in state_dict
      state_dict._metadata = metadata  # type: ignore[attr-defined]

  def load(module, local_state_dict, prefix=''):
      local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
      module._load_from_state_dict(
          local_state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
      for name, child in module._modules.items():
          if child is not None:
              child_prefix = prefix + name + '.'
              child_state_dict = {k: v for k, v in local_state_dict.items() if k.startswith(child_prefix)}
              load(child, child_state_dict, child_prefix)

      # Note that the hook can modify missing_keys and unexpected_keys.
      incompatible_keys = _IncompatibleKeys(missing_keys, unexpected_keys)
      for hook in module._load_state_dict_post_hooks.values():
          out = hook(module, incompatible_keys)
          assert out is None, (
              "Hooks registered with ``register_load_state_dict_post_hook`` are not"
              "expected to return new values, if incompatible_keys need to be modified,"
              "it should be done inplace."
          )

  load(self, state_dict)
  del load

  if strict:
      if len(unexpected_keys) > 0:
          error_msgs.insert(
              0, 'Unexpected key(s) in state_dict: {}. '.format(
                  ', '.join('"{}"'.format(k) for k in unexpected_keys)))
      if len(missing_keys) > 0:
          error_msgs.insert(
              0, 'Missing key(s) in state_dict: {}. '.format(
                  ', '.join('"{}"'.format(k) for k in missing_keys)))

  if len(error_msgs) > 0:
      raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
                          self.__class__.__name__, "\n\t".join(error_msgs)))
  
  if missing_keys:
    warnings.warn(f'missing_keys: {missing_keys}')
  
  if unexpected_keys:
    warnings.warn(f'unexpected_keys: {unexpected_keys}')
  return _IncompatibleKeys(missing_keys, unexpected_keys)


def patch_all():
  warnings.warn('execute PipelineModule.load_state_dir patch')
  from deepspeed.pipe import PipelineModule
  PipelineModule.load_state_dir = load_state_dir
  warnings.warn('execute Module.load_state_dict patch')
#   print(torch.nn.Module._load_state_dict_post_hooks)
#   from torch.nn.modules.module import Module
    
  torch.nn.Module.load_state_dict
#   torch.nn.Module._load_state_dict_post_hooks
#   print(torch.nn.Module._load_state_dict_post_hooks)
#   print(sfsdfg)





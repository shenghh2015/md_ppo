#!/usr/bin/env python

import os
import torch
import json
import sys
from pathlib import Path
import argparse

# insert megatron's root dir into sys.path
root_repo_path = str(Path(__file__).resolve().parents[2])
if root_repo_path not in sys.path:
  sys.path.insert(0, root_repo_path)

# from tools.convert_checkpoint.deepspeed_checkpoint import DeepSpeedCheckpoint
from tools.convert_checkpoint.deepspeed_checkpoint.deepspeed_checkpoint import (
    ARGS_KEY, CHECKPOINT_INFO_KEY, DeepSpeedCheckpoint, SEQUENTIAL_LAYERS,
    LAYER_CONCAT_DIM)
from deepspeed_to_megatron_any_tp_pp import _create_rank_checkpoint, parse_arguments

# the import was tested to work with this version
# https://github.com/huggingface/transformers/commit/0af901e83 if it diverges we may consider
# copying that version here instead
# from transformers.models.megatron_gpt2.convert_megatron_gpt2_checkpoint import convert_megatron_checkpoint
from convert_megatron_bloom_checkpoint import convert_megatron_checkpoint
from transformers import BloomConfig


# rewrite _merge_state_dicts in DeepSpeedCheckpoint
def _merge_state_dicts(self, sd_list):
  merged_sd = {}
  for key in sd_list[0].keys():
    if not key in SEQUENTIAL_LAYERS:
      cat_dim = LAYER_CONCAT_DIM.get(key, 0)
      merged_sd[key] = torch.cat([sd[key] for sd in sd_list], dim=cat_dim)
    else:
      if 'lora_A' in key:
        print(f'avg key: {key}')
        # lora_A 参数需要进行平均
        merged_sd[key] = sum([sd[key] for sd in sd_list], 0.0) / len(sd_list)
      else:
        merged_sd[key] = sd_list[0][key]
  return merged_sd


DeepSpeedCheckpoint._merge_state_dicts = _merge_state_dicts

LAYER_FILE_PREFIX = 'layer_'
MODEL_FILE_PREFIX = 'model_'
EMBEDDING_LAYER_INDEX = 1
TRANSFORMER_LAYER_OFFSET = 3
IGNORE_HF_KEYS = {'lm_head.weight'}  # tied weights in megatron
LORA_FILE_STR = 'lora'

# Bloom ###########
HF_BLOOM_STATE_DICT_MAPPINGS = {
    # ds state dict key => HF state dict key + convert operation
    'word_embeddings.weight': {
        'hf_k': 'transformer.word_embeddings.weight',
    },
    'word_embeddings.norm.weight': {
        'hf_k': 'transformer.word_embeddings_layernorm.weight',
    },
    'word_embeddings.norm.bias': {
        'hf_k': 'transformer.word_embeddings_layernorm.bias',
    },
    'input_layernorm.weight': {
        'hf_k': 'transformer.h.<LAYER>.input_layernorm.weight'
    },
    'input_layernorm.bias': {
        'hf_k': 'transformer.h.<LAYER>.input_layernorm.bias'
    },
    'self_attention.query_key_value.weight': {
        'hf_k': 'transformer.h.<LAYER>.self_attention.query_key_value.weight',
    },
    'self_attention.query_key_value.bias': {
        'hf_k': 'transformer.h.<LAYER>.self_attention.query_key_value.bias',
    },
    'self_attention.query_key_value.lora_A': {
        'hf_k': 'transformer.h.<LAYER>.self_attention.query_key_value.lora_A'
    },
    'self_attention.query_key_value.lora_B': {
        'hf_k': 'transformer.h.<LAYER>.self_attention.query_key_value.lora_B'
    },
    'self_attention.dense.weight': {
        'hf_k': 'transformer.h.<LAYER>.self_attention.dense.weight',
        'row_parallel': True,
    },
    'self_attention.dense.bias': {
        'hf_k': 'transformer.h.<LAYER>.self_attention.dense.bias',
    },
    'post_attention_layernorm.weight': {
        'hf_k': 'transformer.h.<LAYER>.post_attention_layernorm.weight',
    },
    'post_attention_layernorm.bias': {
        'hf_k': 'transformer.h.<LAYER>.post_attention_layernorm.bias',
    },
    'mlp.dense_h_to_4h.weight': {
        'hf_k': 'transformer.h.<LAYER>.mlp.dense_h_to_4h.weight',
    },
    'mlp.dense_h_to_4h.bias': {
        'hf_k': 'transformer.h.<LAYER>.mlp.dense_h_to_4h.bias',
    },
    'mlp.dense_4h_to_h.weight': {
        'hf_k': 'transformer.h.<LAYER>.mlp.dense_4h_to_h.weight',
        'row_parallel': True,
    },
    'mlp.dense_4h_to_h.bias': {
        'hf_k': 'transformer.h.<LAYER>.mlp.dense_4h_to_h.bias',
    },
    'bias': {
        'hf_k': 'transformer.ln_f.bias'
    },
    'weight': {
        'hf_k': 'transformer.ln_f.weight'
    },
}


def nest_tensor_print(d, root=""):
  for k, v in d.items():
    if isinstance(v, dict):
      nest_tensor_print(v, root + f"/{k}")
    else:

      print(root + f"/{k}", v.shape if isinstance(v, torch.Tensor) else v)


def hf_state_to_deepspeed(hf_state_dict,
                          hf_config,
                          tp,
                          deepspeed_checkpoint_dir,
                          dry_run=False):
  hf_sd = hf_state_dict
  checkpoint_dir = deepspeed_checkpoint_dir
  FINAL_LAYER_NORM_INDEX = hf_config.n_layer + TRANSFORMER_LAYER_OFFSET + 1

  if hf_config.model_type == 'bloom':
    original_to_hf_mapping = HF_BLOOM_STATE_DICT_MAPPINGS

  matched_hf_keys = set()

  # Iterate over files in checkpoint_dir
  for fn in sorted(os.listdir(checkpoint_dir)):
    fp = os.path.join(checkpoint_dir, fn)
    print(f'load deepspeed checkpoint: {fp}')
    if os.path.isfile(fp) and fn.endswith('model_states.pt') and fn.startswith(
        LAYER_FILE_PREFIX):
      fn_split = fn.split('-')
      layer_idx = int(fn_split[0][len(LAYER_FILE_PREFIX):])
      model_idx = int(fn_split[1][len(MODEL_FILE_PREFIX):])
      hf_layer_idx = None
      # Determine layer type
      if layer_idx == EMBEDDING_LAYER_INDEX:
        layer_type = 'embedding'
      elif layer_idx == FINAL_LAYER_NORM_INDEX:
        layer_type = 'final_layer_norm'
      else:
        # transformer layer
        hf_layer_idx = layer_idx - TRANSFORMER_LAYER_OFFSET
        layer_type = 'transformer'

      print(
          f'{layer_type=}  {layer_idx} => {hf_layer_idx}       {model_idx=} ')

      # Load state dict from disk to CPU
      sd = torch.load(fp, map_location="cpu")

      for original_k, original_v in sd.items():
        if original_k not in original_to_hf_mapping:
          # if 'lora' in original_k:
          #   continue
          # else:
          raise ValueError(f'There is not mapping for {original_k=}')

        # if original_k in original_to_hf_mapping:
        hf_mapping = original_to_hf_mapping[original_k]
        hf_k = hf_mapping['hf_k']
        # replace layer index
        hf_k = hf_k.replace('<LAYER>', str(hf_layer_idx))

        # get value
        hf_v = hf_sd[hf_k]

        if tp > 1:
          # Tensor parallelism enabled
          if original_v.shape != hf_v.shape:  # no partition when needed

            hf_shape = hf_v.shape
            if 'row_parallel' in hf_mapping and hf_mapping['row_parallel']:
              # row parallel
              single_partition_size = int(hf_shape[1] / tp)
              partition_v = hf_v[:, model_idx *
                                 single_partition_size:(model_idx + 1) *
                                 single_partition_size]

            else:
              # column parallel
              single_partition_size = int(hf_shape[0] / tp)
              partition_v = hf_v[model_idx *
                                 single_partition_size:(model_idx + 1) *
                                 single_partition_size]

            print(
                f' - partitioned from {hf_shape} to {partition_v.shape} ({tp=})'
            )
            hf_v = partition_v

        # check if value shapes match
        if original_v.shape != hf_v.shape:
          raise ValueError(
              f'Shapes do not match: {original_k} = {original_v.shape}; {hf_k} = {hf_v.shape}'
          )

        # check if types are matching
        if original_v.dtype != hf_v.dtype:
          raise ValueError(
              f'Data types do not match: {original_k} = {original_v.dtype}; {hf_k} = {hf_v.dtype}'
          )

        print('matched ', original_k, ' = ', hf_k, '; ')

        matched_hf_keys.add(hf_k)

        # replace in state dict
        sd[original_k] = hf_v

      # save to disk
      if dry_run:
        print('skip saving')
      else:
        torch.save(sd, fp)
        print('saved to ', fp)
        print()

  # Check for not matched keys
  not_matched_hf_keys = set(hf_sd.keys()) - matched_hf_keys - IGNORE_HF_KEYS

  if len(not_matched_hf_keys) > 0:
    raise ValueError('Not matched HF keys: %s' % not_matched_hf_keys)

  print('done')


def _create_latest_file(base_folder, iteration):
  file_path = os.path.join(base_folder, 'latest_checkpointed_iteration.txt')
  os.makedirs(base_folder, exist_ok=True)
  with open(file_path, 'w') as f:
    f.write(str(iteration))


def main():

  # this first part comes mainly from deepspeed_to_megatron.main
  args = parse_arguments()
  print(
      f'Converting DeepSpeed checkpoint in {args.input_folder} to deepspeed checkpoint in {args.output_folder}'
  )
  assert os.path.exists(args.input_folder) and os.path.exists(
      args.output_folder), (os.path.exists(args.input_folder),
                            os.path.exists(args.output_folder))
  # 首先转化成tp=1,pp=1
  print('deepspeed to hf')
  ds_checkpoint = DeepSpeedCheckpoint(args.input_folder, 1, 1)
  iteration = ds_checkpoint.get_iteration()
  # _create_latest_file(args.output_folder, iteration)
  input_state_dict = _create_rank_checkpoint(ds_checkpoint, 0, 0,
                                             args.for_release)

  config = BloomConfig(apply_residual_connection_post_layernorm=False,
                       attention_dropout=0.0,
                       attention_softmax_in_fp32=True,
                       bias_dropout_fusion=True,
                       bos_token_id=1,
                       eos_token_id=2,
                       pad_token_id=3,
                       unk_token_id=0,
                       hidden_dropout=0.0,
                       initializer_range=0.02,
                       layer_norm_epsilon=1e-05,
                       masked_softmax_fusion=True,
                       model_type="bloom",
                       n_embed=2048,
                       n_inner="",
                       n_layer=24,
                       num_attention_heads=16,
                       offset_alibi=100,
                       pretraining_tp=2,
                       seq_length=4096,
                       skip_bias_add=True,
                       skip_bias_add_qkv=False,
                       use_cache=True,
                       vocab_size=250880)

  # Convert.
  print("Converting to HF Checkpoint")

  # nest_tensor_print(input_state_dict)

  output_state_dict = convert_megatron_checkpoint(args, input_state_dict,
                                                  config)

  for k, v in output_state_dict.items():
    print(k, v.shape)

  # 再次转化为deepspeed
  print('hf to deepspeed')
  basename = args.output_folder
  os.makedirs(basename, exist_ok=True)

  hf_state_to_deepspeed(output_state_dict,
                        config,
                        args.target_tp,
                        args.output_folder,
                        dry_run=False)


if __name__ == "__main__":
  main()

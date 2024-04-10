import argparse

import os
import torch

from transformers.models.auto import AutoModelForCausalLM, AutoConfig

LAYER_FILE_PREFIX = 'layer_'
MODEL_FILE_PREFIX = 'model_'
EMBEDDING_LAYER_INDEX = 1
TRANSFORMER_LAYER_OFFSET = 3
IGNORE_HF_KEYS = {'lm_head.weight'}  # tied weights in megatron

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


def merge_transformers_sharded_states(path, num_checkpoints):
  """
    Merge sharded checkpoints from transformers into a single checkpoint.

    Args:
        path (str): the path to the sharded checkpoints
        num_checkpoints (int): the number of checkpoints to merge
    """
  state_dict = {}
  for i in range(1, num_checkpoints + 1):
    checkpoint_path = os.path.join(
        path, f"pytorch_model-{i:05d}-of-{num_checkpoints:05d}.bin")
    print('load hf checkpoint: ', checkpoint_path)
    current_chunk = torch.load(checkpoint_path, map_location="cpu")
    state_dict.update(current_chunk)
  return state_dict


def main():
  """
    Override an existing deepspeed checkpoint with weights from a pretrained HF model.
    Example usage:
    python convert_hf_to_deepspeed.py ${DATASETS_DIR}/huggingface_transformers/pytorch/bloom-1b3 \
        ${EXP_DIR}/tr1/dummy_checkpoints/global_step0 --bf16
    Supported model types: bloom
    :return:
    """
  # Create the argument parser.
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "hf_model_name_or_path",
      type=str,
      help="Path to the pretrained HuggingFace model",
  )
  parser.add_argument(
      "checkpoint_dir",
      type=str,
      help="Path to the DeepSpeed checkpoint directory",
  )
  parser.add_argument(
      "tp",
      type=int,
      default=1,
      help="Tensor parallelism",
  )
  parser.add_argument("--bf16", action="store_true")
  parser.add_argument("--dry-run", action="store_true")
  parser.add_argument("--fp16", action='store_true')
  parser.add_argument("--num_checkpoints", type=int, default=1)
  args = parser.parse_args()

  checkpoint_dir = args.checkpoint_dir
  hf_model_name_or_path = args.hf_model_name_or_path
  bf16 = args.bf16
  dry_run = args.dry_run
  tp = args.tp
  fp16 = args.fp16
  num_checkpoints = args.num_checkpoints

  assert os.listdir(checkpoint_dir), checkpoint_dir

  print(
      f'Loading pretrained HF model from {hf_model_name_or_path} into {checkpoint_dir} ...'
  )

  if not os.path.exists(checkpoint_dir):
    raise FileNotFoundError(
        f'Checkpoint dir does not exists: {checkpoint_dir}')

  if num_checkpoints == 1:
    hf_sd = torch.load(os.path.join(hf_model_name_or_path,
                                    'pytorch_model.bin'),
                       map_location='cpu')
  else:
    hf_sd = merge_transformers_sharded_states(hf_model_name_or_path,
                                              num_checkpoints)

  hf_config = AutoConfig.from_pretrained(hf_model_name_or_path)

  # hf_model = AutoModelForCausalLM.from_pretrained(hf_model_name_or_path,
  # torch_dtype=torch.float16)
  # hf_config = hf_model.config

  if hf_config.model_type == 'bloom':
    original_to_hf_mapping = HF_BLOOM_STATE_DICT_MAPPINGS

  else:
    raise ValueError(f'Unsupported model type: {hf_config.model_typ}')

  FINAL_LAYER_NORM_INDEX = hf_config.n_layer + TRANSFORMER_LAYER_OFFSET + 1

  assert not (fp16 and bf16), f'fp16 and bf16 could only set one'

  # if bf16:
  #   print('Converting HF model to bf16')
  #   hf_model = hf_model.bfloat16()
  if fp16:
    print('Converting HF model to fp16')
    for k in list(hf_sd.keys()):
      hf_sd[k] = hf_sd.pop(k).half()

  # 这里很可能出现key对不上的情况，要根据模型保存的key来修改
  for k in list(hf_sd.keys()):
    new_key = 'transformer.' + k
    hf_sd[new_key] = hf_sd.pop(k)
    print('hf key', new_key)

  # hf_sd = hf_model.state_dict()
  matched_hf_keys = set()

  # Iterate over files in checkpoint_dir
  for fn in sorted(os.listdir(checkpoint_dir)):
    fp = os.path.join(checkpoint_dir, fn)
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
      print(f'open fp: {fp}')
      sd = torch.load(fp, map_location="cpu")

      for original_k, original_v in sd.items():
        if original_k not in original_to_hf_mapping:
          raise ValueError(f'There is not mapping for {original_k=}')

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


if __name__ == "__main__":
  main()

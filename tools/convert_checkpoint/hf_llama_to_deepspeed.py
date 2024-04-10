import argparse

import os
import torch

from transformers.models.auto import AutoModelForCausalLM, AutoConfig

LAYER_FILE_PREFIX = 'layer_'
MODEL_FILE_PREFIX = 'model_'
EMBEDDING_LAYER_INDEX = 1
TRANSFORMER_LAYER_OFFSET = 3
# IGNORE_HF_KEYS = {'lm_head.weight'}  # tied weights in megatron
IGNORE_HF_KEYS = set()

tensor_parallel_params = [
    # megatron-lm layers to merge across tp ranks
    "self_attention.query_key_value.weight",
    "self_attention.query_key_value.bias",
    "self_attention.dense.weight",
    "mlp.dense_h_to_4h.weight",
    "mlp.dense_h_to_4h.bias",
    "mlp.dense_4h_to_h.weight",
    # deprecated
    "attention.query_key_value.weight",
    "attention.query_key_value.bias",
    "attention.dense.weight",
    # transformers layers to split across tp ranks
    "attn.c_attn.weight",
    "attn.c_attn.bias",
    "attn.c_proj.weight",
    "mlp.c_fc.weight",
    "mlp.c_fc.bias",
    "mlp.c_proj.weight",
    'self_attn.q_proj.weight',
    'self_attn.k_proj.weight',
    'self_attn.v_proj.weight',
    'self_attn.o_proj.weight',
    'mlp.down_proj.weight',
    'mlp.up_proj.weight',
    'mlp.gate_proj.weight'
]


# Llama ###########
HF_LLAMA_STATE_DICT_MAPPINGS = {
    # ds state dict key => HF state dict key + convert operation
    'word_embeddings.weight': {
        'hf_k': 'transformer.model.embed_tokens.weight',
    },
    # 'word_embeddings.norm.weight': {
    #     'hf_k': 'transformer.word_embeddings_layernorm.weight',
    # },
    # 'word_embeddings.norm.bias': {
    #     'hf_k': 'transformer.word_embeddings_layernorm.bias',
    # },
    'input_layernorm.weight': {
        'hf_k': 'transformer.model.layers.<LAYER>.input_layernorm.weight'
    },
    # 'input_layernorm.bias': {
    #     'hf_k': 'transformer.model.<LAYER>.input_layernorm.bias'
    # },
    'self_attention.query_key_value.weight': {
        'hf_k': ['transformer.model.layers.<LAYER>.self_attn.q_proj.weight', \
          'transformer.model.layers.<LAYER>.self_attn.k_proj.weight', \
          'transformer.model.layers.<LAYER>.self_attn.v_proj.weight'],
    },
    # 'self_attention.query_key_value.bias': {
    #     'hf_k': 'transformer.model.<LAYER>.self_attention.query_key_value.bias',
    # },
    'self_attention.dense.weight': {
        'hf_k': 'transformer.model.layers.<LAYER>.self_attn.o_proj.weight',
        'row_parallel': True,
    },
    # 'self_attention.dense.bias': {
    #     'hf_k': 'transformer.model.<LAYER>.self_attention.dense.bias',
    # },
    'post_attention_layernorm.weight': {
        'hf_k': 'transformer.model.layers.<LAYER>.post_attention_layernorm.weight',
    },
    # 'post_attention_layernorm.bias': {
    #     'hf_k': 'transformer.model.<LAYER>.post_attention_layernorm.bias',
    # },
    'mlp.dense_h_to_4h.weight': {
        'hf_k': 'transformer.model.layers.<LAYER>.mlp.gate_proj.weight',
    },
    'mlp.dense_h_to_4h_up.weight': {
        'hf_k': 'transformer.model.layers.<LAYER>.mlp.up_proj.weight',
    },
    # 'mlp.dense_h_to_4h.bias': {
    #     'hf_k': 'transformer.model.<LAYER>.mlp.dense_h_to_4h.bias',
    # },
    'mlp.dense_4h_to_h.weight': {
        'hf_k': 'transformer.model.layers.<LAYER>.mlp.down_proj.weight',
        'row_parallel': True,
    },
    # 'mlp.dense_4h_to_h.bias': {
    #     'hf_k': 'transformer.model.<LAYER>.mlp.dense_4h_to_h.bias',
    # },
    # 'bias': {
    #     'hf_k': 'transformer.ln_f.bias'
    # },
    # 'weight': {
    #     'hf_k': 'transformer.ln_f.weight'
    # },
    'self_attention.rotary_emb.inv_freq': {
        'hf_k': 'transformer.model.layers.<LAYER>.self_attn.rotary_emb.inv_freq',
    },
    'weight': {
      'hf_k': 'transformer.model.norm.weight'
    },
    'lm_head.weight': {
      'hf_k': 'transformer.lm_head.weight'
    }
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
  # print(hf_sd)
  # import pdb; pdb.set_trace()
  # hf_model = AutoModelForCausalLM.from_pretrained(hf_model_name_or_path,
  # torch_dtype=torch.float16)
  # hf_config = hf_model.config

  if hf_config.model_type == 'llama':
    original_to_hf_mapping = HF_LLAMA_STATE_DICT_MAPPINGS
    # print(f'model type: llama')

  else:
    raise ValueError(f'Unsupported model type: {hf_config.model_type}')

  FINAL_LAYER_NORM_INDEX = hf_config.num_hidden_layers + TRANSFORMER_LAYER_OFFSET + 1

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
    # new_key = k.replace('transformer.','')
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
        layer_type = 'embed_tokens.weight'
      elif layer_idx == FINAL_LAYER_NORM_INDEX:
        layer_type = 'norm.weight'
      else:
        # transformer layer
        hf_layer_idx = layer_idx - TRANSFORMER_LAYER_OFFSET
        layer_type = 'transformer layers'

      print(
          f'{layer_type=}  {layer_idx} => {hf_layer_idx}       {model_idx=} ')

      # Load state dict from disk to CPU
      print(f'open fp: {fp}')
      sd = torch.load(fp, map_location="cpu")

      qkv_weight_to_combine = {}
      mlp_weight_to_combine = {}
      for original_k, original_v in sd.items():
        if original_k not in original_to_hf_mapping:
          # raise ValueError(f'There is not mapping for {original_k=}')
          continue

        hf_mapping = original_to_hf_mapping[original_k]
        hf_k = hf_mapping['hf_k']
        tmp_hf_k = []
        # replace layer index
        if isinstance(hf_k, list):
          for op_name in hf_k:
            op_name = op_name.replace('<LAYER>', str(hf_layer_idx))
            tmp_hf_k.append(op_name)
            # print(f'op_name is: {op_name}')
            if 'self_attn' in op_name:
              # print(f'op name is: {op_name}')
              # print(hf_layer_idx)
              n_head = 32
              if 'q_proj' in op_name:
                  hidden_dim = hf_sd[op_name].shape[-1]
                  qkv_weight_to_combine['q_proj'] = hf_sd[op_name].reshape(n_head, -1, hidden_dim)
              elif 'k_proj' in op_name:
                  qkv_weight_to_combine['k_proj'] = hf_sd[op_name].reshape(n_head, -1, hidden_dim)
              elif 'v_proj' in op_name:
                  qkv_weight_to_combine['v_proj'] = hf_sd[op_name].reshape(n_head, -1, hidden_dim)

              if len(qkv_weight_to_combine) == 3:
                q_weights = qkv_weight_to_combine['q_proj'].chunk(tp, dim=0)
                k_weights = qkv_weight_to_combine['k_proj'].chunk(tp, dim=0)
                v_weights = qkv_weight_to_combine['v_proj'].chunk(tp, dim=0)
                result_weights = []
                for idx in range(len(q_weights)):
                    partition_weight = torch.cat([q_weights[idx], k_weights[idx], v_weights[idx]], dim = 1)
                    hidden_dim = partition_weight.shape[-1] 
                    partition_weight = partition_weight.reshape(-1, hidden_dim)
                    result_weights.append(partition_weight)

                hf_v = torch.cat(result_weights)
                print(f'hf_v shape:{hf_v.shape}')
                # layer_name = f"layers.{layer}.self_attention.query_key_value.{weight_or_bias}"
              else:
                  continue
            elif 'mlp' in op_name:
              if 'gate_proj' in op_name:
                  assert (len(mlp_weight_to_combine) != 2)
                  mlp_weight_to_combine['gate_proj'] = hf_sd[op_name]
              elif 'up_proj' in op_name:
                  assert (len(mlp_weight_to_combine) != 2)
                  mlp_weight_to_combine['up_proj'] = hf_sd[op_name]

              if 'down_proj' not in op_name and len(mlp_weight_to_combine) == 2:
                  gate_weights = mlp_weight_to_combine['gate_proj'].chunk(tp, dim=0)
                  up_weights = mlp_weight_to_combine['up_proj'].chunk(tp, dim=0)
                  result_weights = []
                  for idx in range(len(gate_weights)):
                      partition_weight = torch.cat([gate_weights[idx], up_weights[idx]])
                      result_weights.append(partition_weight)

                  hf_v = torch.cat(result_weights)
                  # layer_name = f"layers.{layer}.mlp.dense_h_to_4h.{weight_or_bias}"
              elif 'down_proj' not in op_name and len(mlp_weight_to_combine) < 2:
                  continue
          hf_k = tmp_hf_k
        else: 
          hf_k = hf_k.replace('<LAYER>', str(hf_layer_idx))
          # print(hf_k)
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

        if isinstance(hf_k, list):
          for single_key in hf_k:
            matched_hf_keys.add(single_key)
        else:
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

import argparse

import os
import torch
import sys
import json
from collections import OrderedDict


def merge_and_split_weight(state_dicts, key, tp_index, tp_total):
  weights = [s[key] for s in state_dicts]
  if key in COLUMN_PARALLEL_KEYS:
    merged_weights = torch.cat(weights, dim=0)
    tp_dim = merged_weights.shape[0] // tp_total
    splited_weights = merged_weights[tp_index * tp_dim:(tp_index + 1) *
                                     tp_dim, :]
  elif key in ROW_PARALLEL_KEYS:
    merged_weights = torch.cat(weights, dim=1)
    tp_dim = merged_weights.shape[-1] // tp_total
    splited_weights = merged_weights[:,
                                     tp_index * tp_dim:(tp_index + 1) * tp_dim]
  elif key == EMBEDDING_KEY:
    merged_weights = torch.cat(weights, dim=1)
    tp_dim = merged_weights.shape[0] // tp_total
    splited_weights = merged_weights[tp_index * tp_dim:(tp_index + 1) *
                                     tp_dim, :]
  else:
    assert key in NORM_KEYS, "invalid key"
    splited_weights = state_dicts[0][key]

  return splited_weights


def _save_checkpoint(file_path, chkpt_sd):
  dir, _ = os.path.split(file_path)
  os.makedirs(dir, exist_ok=True)
  torch.save(chkpt_sd, file_path)


def _create_checkpoint_paths(base_folder, iteration, tp_degree, pp_degree):
  path_list = []
  iter_folder = f'iter_{iteration:07d}'
  for i in range(0, tp_degree):
    path_list.append([])
    for j in range(0, pp_degree):
      rank_folder = f'mp_rank_{i:02d}' if pp_degree == 1 else f'mp_rank_{i:02d}_{j:03d}'
      ckpt_path = os.path.join(rank_folder, 'model_optim_rng.pt')
      path_list[i].append(os.path.join(base_folder, iter_folder, ckpt_path))

  return path_list


def _create_rank_checkpoint(state_dicts, tp_index, pp_index, tp_total,
                            pp_total, total_layers, iteration):
  megatron_state_dict = OrderedDict()
  megatron_state_dict['iteration'] = iteration
  megatron_state_dict['model'] = OrderedDict()
  layer_ids = list(
      range(pp_index * (total_layers // pp_total), \
          (pp_index + 1) * (total_layers // pp_total)
      )
  )

  if pp_index == 0:
    # add embedding
    megatron_state_dict['model']['word_embeddings.weight'] = \
        merge_and_split_weight(
            state_dicts, EMBEDDING_KEY, tp_index, tp_total
        )

  for layer_id in layer_ids:
    for key_pattern in LAYER_KEYS:
      key = key_pattern % layer_id
      megatron_state_dict['model'][key] = merge_and_split_weight(
          state_dicts, key, tp_index, tp_total)

  if pp_index == pp_total - 1:
    # add output layer
    for key in OUTPUT_KEYS:
      megatron_state_dict['model'][key] = merge_and_split_weight(
          state_dicts, key, tp_index, tp_total)
  return megatron_state_dict


# shape is (X // pp, Y)
COLUMN_PARALLEL_KEYS = set([
    "layers.%d.attention.wq.weight",
    "layers.%d.attention.wk.weight",
    "layers.%d.attention.wv.weight",
    "layers.%d.feed_forward.w1.weight",
    "layers.%d.feed_forward.w3.weight",
    "output.weight",
])

# shape is (X, Y // pp)
ROW_PARALLEL_KEYS = set(
    ["layers.%d.attention.wo.weight", "layers.%d.feed_forward.w2.weight"])

# shape is (hidden_dim, ), shared by models of different ranks
NORM_KEYS = set([
    "layers.%d.attention_norm.weight", "layers.%d.ffn_norm.weight",
    "norm.weight"
])
# shape is (X, Y // tp)
EMBEDDING_KEY = "tok_embeddings.weight"

LAYER_KEYS = set([
    "layers.%d.attention.wq.weight", "layers.%d.attention.wk.weight",
    "layers.%d.attention.wv.weight", "layers.%d.attention.wo.weight",
    "layers.%d.feed_forward.w1.weight", "layers.%d.feed_forward.w2.weight",
    "layers.%d.feed_forward.w3.weight", "layers.%d.attention_norm.weight",
    "layers.%d.ffn_norm.weight"
])

OUTPUT_KEYS = set(["norm.weight", "output.weight"])


def expand_keys(keys, total_layers):
  expand_keys = []
  for key in keys:
    if "%d" in key:
      for i in range(total_layers):
        expand_keys.append(key % i)
    else:
      expand_keys.append(key)
  return set(expand_keys)


def main():
  """
    Override an existing deepspeed checkpoint with weights from a pretrained HF model.
    Example usage:
    python convert_hf_to_deepspeed.py ${DATASETS_DIR}/huggingface_transformers/pytorch/bloom-1b3 \
        ${EXP_DIR}/tr1/dummy_checkpoints/global_step0 --bf16
    Supported model types: llama
    :return:
    """
  # Create the argument parser.
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--input_dir",
      type=str,
      help="Path to the pretrained HuggingFace model",
  )
  parser.add_argument(
      "--output_dir",
      type=str,
      help="Path to the DeepSpeed checkpoint directory",
  )
  parser.add_argument(
      "--tp",
      type=int,
      default=1,
      help="Tensor parallelism",
  )
  parser.add_argument(
      "--pp",
      type=int,
      default=1,
      help="Pipeline parallelism",
  )

  parser.add_argument(
      "--iteration",
      type=int,
      default=10000,
      help="Pipeline parallelism",
  )
  parser.add_argument("--bf16", action="store_true")
  parser.add_argument("--dry-run", action="store_true")
  parser.add_argument("--fp16", action='store_true')
  args = parser.parse_args()

  output_dir = args.output_dir
  input_dir = args.input_dir
  bf16 = args.bf16
  dry_run = args.dry_run
  tp = args.tp
  pp = args.pp
  fp16 = args.fp16
  iteration = args.iteration

  global COLUMN_PARALLEL_KEYS, ROW_PARALLEL_KEYS, NORM_KEYS

  os.makedirs(output_dir, exist_ok=True)

  checkpoints = [
      f for f in sorted(os.listdir(input_dir)) if f.endswith('.pth')
  ]
  # Iterate over files in output_dir
  state_dicts = []
  model_param = json.load(open(os.path.join(input_dir, "params.json")))
  checkpoint_paths = _create_checkpoint_paths(output_dir, iteration, tp, pp)
  print(checkpoint_paths)
  for fn in checkpoints:
    fp = os.path.join(input_dir, fn)
    state_dicts.append(torch.load(fp, map_location="cpu"))
  total_layers = model_param["n_layers"]

  COLUMN_PARALLEL_KEYS = expand_keys(COLUMN_PARALLEL_KEYS, total_layers)
  ROW_PARALLEL_KEYS = expand_keys(ROW_PARALLEL_KEYS, total_layers)
  NORM_KEYS = expand_keys(NORM_KEYS, total_layers)

  assert total_layers % pp == 0, "total_layers is not divisible by pp"
  for i in range(0, tp):
    for j in range(0, pp):
      print(f'正在转换: tp={i}, pp: {j}')

      state_dict = _create_rank_checkpoint(state_dicts, i, j, tp, pp,
                                           total_layers, iteration)
      _save_checkpoint(checkpoint_paths[i][j], state_dict)
  with open(os.path.join(output_dir, "latest_checkpointed_iteration.txt"),
            "w") as f:
    f.write(str(iteration))


if __name__ == "__main__":
  main()

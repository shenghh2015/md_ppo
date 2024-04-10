import os
import torch
import re
import argparse


from tools.convert_checkpoint.megatron_to_transformers import \
  _convert_checkpoint_from_megatron_to_transformers_state_dict,\
  get_element_from_dict_by_path,\
  transformers_to_megatron_fix_query_key_value_ordering,\
  transformers_to_megatron,\
  tensor_parallel_params,\
  recursive_print

from megatron.tokenizer.tokenizer import _vocab_size_with_padding
import pdb

def convert_checkpoint_megatron_to_megatron(
    input_megatron_path,
    output_megatron_path,
    target_megatron_tp=1,
    target_megatron_pp=None,
    custom_layer_split=None,
    iteration=0,
):
  """
  Args:
      input_megatron_path (_type_): _description_
      output_megatron_path (_type_): _description_
  """
  assert not (target_megatron_pp is not None and custom_layer_split is not None), \
     (target_megatron_pp, custom_layer_split)

  if custom_layer_split is not None:
    assert isinstance(custom_layer_split, list), custom_layer_split
    assert custom_layer_split, custom_layer_split

  os.makedirs(output_megatron_path, exist_ok=True)
  config, margs, state_dict = _convert_checkpoint_from_megatron_to_transformers_state_dict(
      input_megatron_path)
  recursive_print('input_state', state_dict)
  if target_megatron_pp is not None:
    if config.num_hidden_layers % target_megatron_pp != 0:
      raise ValueError(
          f"Number of layers ({config.num_hidden_layers}) must be divisible by number of tensor parallelism"
          f" ({target_megatron_pp})")
    num_layers = config.num_hidden_layers // target_megatron_pp
    layer_splits = [num_layers for _ in range(target_megatron_pp)]
  else:
    # 保证layer的总和等于num_hedden_layers
    assert sum(
        custom_layer_split) == config.num_hidden_layers, custom_layer_split
    target_megatron_pp = len(custom_layer_split)
    layer_splits = custom_layer_split

  # for k,v in state_dict.items():
  #   if isinstance(v,dict):
  #     print(k,v)
  #   else:

  #     print(k,v.shape)

  # print(sdg)

  # Saving the tracker file
  tracker_filepath = os.path.join(output_megatron_path,
                                  "latest_checkpointed_iteration.txt")
  with open(tracker_filepath, "w") as f:
    f.write(str(iteration))

  iter_folder = f'iter_{iteration:07d}'

  dtype = torch.float16
  setattr(margs, "params_dtype", dtype)
  # Convert.
  print("Converting")  #
  # pp等于1的时候需要加上embedding
  output_state_dict = []
  for i in range(target_megatron_tp):
    output_state_dict.append({})

  # Embedding layer
  print("converting embedding layer")
  # pos_embedding = state_dict["transformer.wpe.weight"].to(dtype)
  word_embedding = state_dict["transformer.word_embeddings.weight"].to(dtype)
  orig_vocab_size = config.vocab_size
  padded_vocab_size = _vocab_size_with_padding(orig_vocab_size, margs)
  setattr(margs, "padded_vocab_size", padded_vocab_size)

  setattr(margs, 'tensor_model_parallel_size', target_megatron_tp)
  setattr(margs, 'pipeline_model_parallel_size', target_megatron_pp)

  # Cut out extra padding we don't need
  if orig_vocab_size > padded_vocab_size:
    full_word_embed = word_embedding[0:padded_vocab_size, :]
  # Expanding embedding to larger size by replicating final entry
  elif orig_vocab_size < padded_vocab_size:
    padding_size = padded_vocab_size - orig_vocab_size
    full_word_embed = torch.cat(
        (word_embedding,
         word_embedding[-1].unsqueeze(0).expand(padding_size, -1)))
  # Same size!
  else:
    full_word_embed = word_embedding

  # Split into new tensor model parallel sizes
  out_word_embed = torch.chunk(full_word_embed, target_megatron_tp, dim=0)
  for i in range(target_megatron_tp):
    # pos_emb_dict = get_element_from_dict_by_path(
    #     output_state_dict[i], "model.language_model.embedding.position_embeddings"
    # )
    # pos_emb_dict["weight"] = pos_embedding

    word_emb_dict = get_element_from_dict_by_path(
        output_state_dict[i], "model.language_model.embedding.word_embeddings")
    word_emb_dict["weight"] = out_word_embed[i]
  # 增加layer norm参数
  # word_emb_dict["norm.weight"] = state_dict[
  #     "transformer.word_embeddings.norm.weight"].to(dtype)
  # word_emb_dict["norm.bias"] = state_dict[
  #     "transformer.word_embeddings.norm.bias"].to(dtype)
  norm_weight = state_dict[
      "transformer.word_embeddings.norm.weight"].to(dtype)
  norm_bias = state_dict[
      "transformer.word_embeddings.norm.bias"].to(dtype)
  # print(f"******************")
  # print(f"norm.weight:{norm_weight.size()} and norm.bias:{norm_bias.size()}")
  # norm_weight = torch.chunk(norm_weight, target_megatron_tp, dim=0)
  # norm_bias = torch.chunk(norm_bias, target_megatron_tp, dim=0)
  for i in range(target_megatron_tp):
    norm_dict = get_element_from_dict_by_path(
        output_state_dict[i], "model.language_model.embedding.word_embeddings")
    norm_dict['norm.weight'] = norm_weight
    norm_dict['norm.bias'] = norm_bias
  # Transformer layers
  print("converting transformer layers")

  # if config.num_hidden_layers % target_megatron_pp != 0:
  #   raise ValueError(
  #       f"Number of layers ({config.num_hidden_layers}) must be divisible by number of tensor parallelism"
  #       f" ({target_megatron_pp})")
  # num_layers = config.num_hidden_layers // target_megatron_pp
  layer_re = re.compile(r"transformer.h\.(\d+)\.([a-z0-9_.]+)\.([a-z_A-z]+)")
  # layer_re = re.compile("layers\.(\d+)\.([a-z0-9_.]+)\.([a-z_A-z]+)")
  # The number of heads.
  heads = config.n_head
  # The hidden_size per head.
  hidden_size_per_head = config.n_embd // config.n_head

  for pp_rank in range(target_megatron_pp):
    if pp_rank == 0:
      layer_offset = 0
    else:
      layer_offset = sum(layer_splits[:pp_rank])

    if pp_rank > 0:
      output_state_dict = []
      for i in range(target_megatron_tp):
        output_state_dict.append({})

    for layer in range(layer_splits[pp_rank]):
      pp_layer_id = layer + layer_offset
      layers_to_copy = [
          layer_name for layer_name in state_dict.keys()
          if layer_name.startswith(f"transformer.h.{pp_layer_id}.")
      ]

      for layer_name in layers_to_copy:
        m = layer_re.match(layer_name)
        # Stop if that's not a layer
        if m is None:
          break

        # The index of the layer.
        _ = int(m.group(1))
        # The name of the operation.
        op_name = m.group(2)
        # Is it a weight or a bias?
        weight_or_bias = m.group(3)

        params = state_dict[layer_name].to(dtype)

        # print('op_name/weight_or_bias',op_name,weight_or_bias)
        # handle layernorm
        if op_name in ['input_layernorm', 'post_attention_layernorm']:
          layer_name = f"layers.{layer}.{op_name}.{weight_or_bias}"

        # handle attention K, V, Q weights
        elif op_name.startswith(
            "self_attention.query_key_value") and weight_or_bias == "weight":
          # transformers stores D X (3*D) but Megatron-LM expects (3*D) X D.
          # params = params.transpose(0, 1).contiguous()

          params = transformers_to_megatron_fix_query_key_value_ordering(
              params,
              3.0,
              3,
              heads,
              hidden_size_per_head,
          )
          layer_name = f"layers.{layer}.self_attention.query_key_value.{weight_or_bias}"

        # handle attention K, V, Q bias
        # self_attention.query_key_value.bias
        elif op_name.startswith(
            "self_attention.query_key_value") and weight_or_bias == "bias":
          params = transformers_to_megatron_fix_query_key_value_ordering(
              params,
              3.0,
              3,
              heads,
              hidden_size_per_head,
          )
          layer_name = f"layers.{layer}.self_attention.query_key_value.{weight_or_bias}"

        elif (op_name == "attention.query_key_value"
              or op_name == "self_attention.query_key_value") and (
                  weight_or_bias in ["lora_A", "lora_B"]):
          layer_name = f"layers.{layer}.self_attention.query_key_value.{weight_or_bias}"

        # handle attention and mlp weights
        elif weight_or_bias == "weight":
          out_name = transformers_to_megatron.get(op_name, None)
          if out_name is None:
            continue
          layer_name = f"layers.{layer}.{out_name}.{weight_or_bias}"

        # handle attention and mlp bias
        elif weight_or_bias == "bias":
          out_name = transformers_to_megatron.get(op_name, None)
          if out_name is None:
            continue
          layer_name = f"layers.{layer}.{out_name}.{weight_or_bias}"

        # skip
        else:
          continue

        # 处理张量并行
        if op_name + "." + weight_or_bias in tensor_parallel_params:
          # dim = 1 if op_name in ["attn.c_proj", "mlp.c_proj"] else 0
          dim = 1 if op_name in [
              "self_attention.dense", "mlp.dense_4h_to_h", "attention.dense"
          ] else 0
          params = torch.chunk(params, target_megatron_tp, dim=dim)

        for i in range(target_megatron_tp):
          params_dict = get_element_from_dict_by_path(
              output_state_dict[i], "model.language_model.encoder")
          params_dict[layer_name] = (params[i] if
                                     (op_name + "." + weight_or_bias
                                      in tensor_parallel_params) else params)

    if pp_rank == target_megatron_pp - 1:
      # handle final layernorm
      for weight_or_bias in ["weight", "bias"]:
        params = state_dict[f"transformer.ln_f.{weight_or_bias}"].to(dtype)
        layer_name = f"final_layernorm.{weight_or_bias}"
        for i in range(target_megatron_tp):
          params_dict = get_element_from_dict_by_path(
              output_state_dict[i], "model.language_model.encoder")
          params_dict[layer_name] = params

      # 处理v_head
      v_head_state_dict = {}
      for k, v in state_dict.items():
        if k.startswith('v_head.'):
          v_head_state_dict[k.replace('v_head.', '')] = v

      if v_head_state_dict:
        for v_layer_name in v_head_state_dict:
          if 'dense.layer.weight' in v_layer_name:
            params = v_head_state_dict[v_layer_name].to(dtype)
            params = torch.chunk(params, target_megatron_tp, dim=-1)
            for i in range(target_megatron_tp):
              params_dict = get_element_from_dict_by_path(
                  output_state_dict[i], "model.v_head")
              params_dict[v_layer_name] = params[i]
          else:
            params = v_head_state_dict[v_layer_name].to(dtype)
            for i in range(target_megatron_tp):
              params_dict = get_element_from_dict_by_path(
                  output_state_dict[i], "model.v_head")
              params_dict[v_layer_name] = params
        # weight
        
        # bias

        # others
        # output_state_dict[i]['model']['v_head'] = v_head_state_dict

      # if 'v_head.layer.weight' in state_dict:
      #   # v_head 写入
      #   assert target_megatron_tp <=1, 'not to support'
      #   # print(output_state_dict[0].keys())
      #   output_state_dict[0]['model']['v_head'] = {}
      #   output_state_dict[0]['model']['v_head']['layer.weight'] = state_dict['v_head.layer.weight']
      #   output_state_dict[0]['model']['v_head']['layer.bias'] = state_dict['v_head.layer.bias']

      # if 'v_head_layernorm' in

      # add the LM head
      for i in range(target_megatron_tp):
        params_dict = get_element_from_dict_by_path(
            output_state_dict[i], "model.word_embeddings_for_head")
        params_dict["weight"] = out_word_embed[i]

    # saving the state dict as per the tp_rank and pp_rank
    for tp_rank in range(target_megatron_tp):
      output_state_dict[tp_rank]["checkpoint_version"] = 3.0
      output_state_dict[tp_rank]["args"] = margs
      checkpoint_dir = (f"mp_rank_{tp_rank:02d}" if target_megatron_pp == 1
                        else f"mp_rank_{tp_rank:02d}_{pp_rank:03d}")
      # if use_distributed_optimizer:
      #   checkpoint_name = "model_rng.pt"
      # else:
      checkpoint_name = "model_optim_rng.pt"
      #   output_state_dict[tp_rank]["optimizer"] = dummy_optim_state_dict[
      #       "optimizer"]
      checkpoint_dir = os.path.join(output_megatron_path, iter_folder,
                                    checkpoint_dir)
      os.makedirs(checkpoint_dir, exist_ok=True)
      checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
      # if args.print_checkpoint_structure:
      print(
          f"Checkpoint structure of model state dict shard belonging to TP rank {tp_rank} and PP rank"
          f" {pp_rank}:")
      recursive_print(None, output_state_dict[tp_rank])
      print('save to ', checkpoint_path)
      torch.save(output_state_dict[tp_rank], checkpoint_path)


def parse_args():
  parser = argparse.ArgumentParser("megatron to megatron")
  parser.add_argument("--input_megatron_path",
                      type=str,
                      help="input megatron path")
  parser.add_argument("--output_megatron_path",
                      type=str,
                      help="output megatron path")
  parser.add_argument("--target_tp", type=int, help="megatron target tp")
  parser.add_argument("--target_pp",
                      type=int,
                      default=None,
                      help="megatron target pp")
  parser.add_argument(
      "--custom_layer_split",
      type=lambda x: eval(x),
      default=None,
      help=
      "custom layer split, if target_pp is None, the uniform distribution is considered"
  )

  return parser.parse_args()


def main():
  args = parse_args()
  convert_checkpoint_megatron_to_megatron(
      args.input_megatron_path,
      args.output_megatron_path,
      target_megatron_tp=args.target_tp,
      target_megatron_pp=args.target_pp,
      custom_layer_split=args.custom_layer_split)


if __name__ == "__main__":
  main()

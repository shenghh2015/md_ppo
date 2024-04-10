import os
import re
import torch
import json
import sys
import argparse

from pathlib import Path
from collections import OrderedDict
# insert megatron's root dir into sys.path
root_repo_path = str(Path(__file__).resolve().parents[2])
if root_repo_path not in sys.path:
  sys.path.insert(0, root_repo_path)

# from deepspeed.checkpoint import DeepSpeedCheckpoint
from megatron.tokenizer.tokenizer import _vocab_size_with_padding
from deepspeed_to_megatron_helper import _create_rank_checkpoint, parse_arguments
from tools.convert_checkpoint.deepspeed_checkpoint.deepspeed_checkpoint_wo_zeros import DeepSpeedCheckpoint

# the import was tested to work with this version
# https://github.com/huggingface/transformers/commit/0af901e83 if it diverges we may consider
# copying that version here instead
# from transformers.models.megatron_gpt2.convert_megatron_gpt2_checkpoint import convert_megatron_checkpoint
from convert_megatron_bloom_checkpoint import convert_megatron_checkpoint
from transformers import GPT2Config, BloomConfig

from tools.convert_checkpoint.megatron_to_transformers import \
  _convert_checkpoint_from_megatron_to_transformers_state_dict,\
  get_element_from_dict_by_path,\
  transformers_to_megatron_fix_query_key_value_ordering,\
  transformers_to_megatron,\
  tensor_parallel_params,\
  recursive_print
  
from tools.convert_checkpoint.deepspeed_checkpoint.deepspeed_checkpoint import (
    ARGS_KEY,
    CHECKPOINT_INFO_KEY,
    # DeepSpeedCheckpoint,
    SEQUENTIAL_LAYERS,
    LAYER_CONCAT_DIM
)
LAYER_FILE_PREFIX = 'layer_'
LAYER_FILE_INDEX_PATTERN = '_00-'
MODEL_FILE_PREFIX = 'model_'
LAYER_CONCAT_DIM = {'self_attention.dense.weight': 1, 'mlp.dense_4h_to_h.weight': 1}
MODEL_KEY = 'model'
ARGS_KEY = 'args'
LANGUGAGE_MODEL_KEY = 'language_model'
EMBEDDING_KEY = 'embedding'
ENCODER_KEY = 'encoder'
WORD_EMBEDDINGS_FOR_HEAD_KEY = 'word_embeddings_for_head'
WORD_EMBEDDINGS_KEY = 'word_embeddings'
FINAL_LAYER_NORM_KEY = 'final_layernorm'
CHECKPOINT_VERSION_KEY = 'checkpoint_version'
CHECKPOINT_VERSION_VALUE = 3.0
ITERATION_KEY = 'iteration'
  
def _list_type(s):
  r = eval(s)
  if not isinstance(r,list):
    r = eval(r)
  assert isinstance(r,list), s

  return r

def parse_arguments():
  parser = argparse.ArgumentParser()
  parser.add_argument('--input_folder',
                      default=None,
                      type=str,
                      help='Input DeepSpeed Checkpoint folder')
  parser.add_argument('--output_folder',
                      default=None,
                      type=str,
                      help='Output Megatron checkpoint folder')
  parser.add_argument('--target_tp',
                      default=1,
                      type=int,
                      help='Target TP degree')
  parser.add_argument('--target_pp',
                      default=None,
                      type=int,
                      help='Target PP degree')
  parser.add_argument(
    "--custom_layer_split", 
    type=_list_type,
    default=None,
    help="custom layer split, if target_pp is None, the uniform distribution is considered"
    )
  parser.add_argument('--no_sync_tp_duplicated_parameters',
                      action='store_true',
                      help='Target PP degree')
  parser.add_argument(
      '--for_release',
      action='store_true',
      help='Convert for release purpose, reset some (progress) counters.')
  args = parser.parse_args()
  print(f'args = {args}')
  return args


def _merge_state_dicts(self, sd_list):
    merged_sd = {}
    for key in sd_list[0].keys():
        if not key in SEQUENTIAL_LAYERS:
            cat_dim = LAYER_CONCAT_DIM.get(key, 0)
            merged_sd[key] = torch.cat([sd[key] for sd in sd_list], dim=cat_dim)
        else:
            if not self.no_sync_tp_duplicated_parameters or 'lora_A' in key:
              # print(f'avg key: {key}')
              merged_sd[key] = sum([sd[key] for sd in sd_list],0.0)/len(sd_list)
            else:
              merged_sd[key] = sd_list[0][key]

    return merged_sd
  
DeepSpeedCheckpoint._merge_state_dicts = _merge_state_dicts

def _convert_ds_transformer_state(sd_list):
   new_sd = OrderedDict()
   for i, sd in enumerate(sd_list):
     for key, value in sd.items():
       new_key = f'layers.{i}.{key}'
       new_sd[new_key] = value

   return new_sd
 

def _create_megatron_dict():
  language_model_dict = {EMBEDDING_KEY: {}, ENCODER_KEY: {}}
  megatron_dict = {
      MODEL_KEY: {
          LANGUGAGE_MODEL_KEY: language_model_dict
      },
      CHECKPOINT_VERSION_KEY: CHECKPOINT_VERSION_VALUE
  }
  return megatron_dict

def pretty(d, indent=0):
   for key, value in d.items():
     print('\t' * indent + str(key))
     if isinstance(value, dict):
       pretty(value, indent + 1)
     else:
       print('\t' * (indent + 1) + str(value.shape))

def _renest_sd(sd):
   # print(sd)
   new_sd = OrderedDict()
   for key, value in sd.items():
     # print('fix wordembedding',key,value.shape)
     key_list = key.split('.')
     if len(key_list) == 2:
       a, b = key_list
       new_sd[a] = {b: value}
     else:
       # 处理有embedding layer norm的情况
       # print('embedding norm', key, value.shape)
       assert len(key_list) == 3
       if key_list[0] not in new_sd:
         new_sd[key_list[0]] = {}
       if key_list[1] not in new_sd[key_list[0]]:
         new_sd[key_list[0]][key_list[1]] = {}

       # new_sd[key_list[0]][key_list[1]].update(
       #   **{
       #     key_list[2]:value
       #   }
       # )

       new_sd[key_list[0]][f'{key_list[1]}.{key_list[2]}'] = value
       # print(f'keys :{new_sd[key_list[0]][key_list[1]]}')
       # new_sd[key_list[0]].update(**{
       #     key_list[1]:{
       #         key_list[2]:value
       #     }
       # })
     # a = '.'.join(key_list[:-1])
     # b = key_list[-1]

   pretty(new_sd)

   return new_sd

def _create_rank_checkpoint(ds_checkpoint,
                             # checkpoint_path,
                             tp_index,
                             pp_index,
                             for_release=False):
    meg_encoder_sd = OrderedDict()
    meg_embedding_sd = OrderedDict()
    meg_embedding_for_head_sd = OrderedDict()

    transformer_sd = ds_checkpoint.get_transformer_state(tp_index, pp_index)
    meg_encoder_sd.update(_convert_ds_transformer_state(transformer_sd))

    if pp_index in [0, ds_checkpoint.pp_degree - 1]:
      embedding_sd = ds_checkpoint.get_embedding_state(tp_index)
      for k,v in embedding_sd.items():
        print(k, v.shape)
      nested_embedding_sd = _renest_sd(embedding_sd)
      if pp_index == 0:
        meg_embedding_sd.update(nested_embedding_sd)
      # 处理最后一个stage
      if pp_index == ds_checkpoint.pp_degree - 1:
        for key, value in embedding_sd.items():
          if key.startswith(WORD_EMBEDDINGS_KEY):
            fields = key.split('.')
            new_fields = fields[1:]
            new_key = '.'.join(new_fields)
            # 最后一个stage不进行norm 参数的写入
            # ADD by zx
            if 'norm' not in new_key:
              print(f'pp: {pp_index} write key: {new_key}, {value.shape}')
              meg_embedding_for_head_sd[new_key] = value

        final_norm_sd = ds_checkpoint.get_final_norm_state(tp_index)
        new_final_norm_sd = {
            f'{FINAL_LAYER_NORM_KEY}.{key}': value
            for key, value in final_norm_sd.items()
        }
        meg_encoder_sd.update(new_final_norm_sd)

    # pretty(meg_encoder_sd)
    # pretty(meg_embedding_for_head_sd)
    checkpoint_sd = _create_megatron_dict()

    iteration = ds_checkpoint.get_iteration()
    checkpoint_sd[ITERATION_KEY] = iteration

    # pp=1的时候也写入word embedding
    if pp_index == 0:
      checkpoint_sd[MODEL_KEY][LANGUGAGE_MODEL_KEY][
          EMBEDDING_KEY] = meg_embedding_sd
    checkpoint_sd[MODEL_KEY][LANGUGAGE_MODEL_KEY][ENCODER_KEY] = meg_encoder_sd

    # 最后一个pp写入word embedding
    if pp_index == ds_checkpoint.pp_degree - 1:
      checkpoint_sd[MODEL_KEY][
          WORD_EMBEDDINGS_FOR_HEAD_KEY] = meg_embedding_for_head_sd

    checkpoint_sd[ARGS_KEY] = ds_checkpoint.get_args()
    # Adjust specific fields
    checkpoint_sd[ARGS_KEY].tensor_model_parallel_size = ds_checkpoint.tp_degree
    checkpoint_sd[
        ARGS_KEY].pipeline_model_parallel_size = ds_checkpoint.pp_degree
    if for_release:
      checkpoint_sd[ARGS_KEY].consumed_train_samples = 0
      checkpoint_sd[ARGS_KEY].consumed_valid_samples = 0

    return checkpoint_sd
       

def convert_deepspeed_to_hf(args):
  print(
      f'Converting DeepSpeed checkpoint in {args.input_folder} to HF Transformers checkpoint in {args.output_folder}'
  )

  ds_checkpoint = DeepSpeedCheckpoint(args.input_folder, 1,
                                      1)
  
  ds_checkpoint.no_sync_tp_duplicated_parameters = args.no_sync_tp_duplicated_parameters
  megatron_args = ds_checkpoint.get_args()
  iteration = ds_checkpoint.get_iteration()
  input_state_dict = _create_rank_checkpoint(ds_checkpoint, 0, 0,
                                             args.for_release)

  config = BloomConfig(
    apply_residual_connection_post_layernorm=False,
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
    n_layer=30,
    num_attention_heads=32,
    offset_alibi=100,
    pretraining_tp=2,
    seq_length=4096,
    skip_bias_add=True,
    skip_bias_add_qkv=False,
    use_cache=True,
    vocab_size=250880
  )

  # Convert.
  print("Converting to HF Checkpoint")
  output_state_dict = convert_megatron_checkpoint(args, input_state_dict,
                                                  config)
  for key in list(output_state_dict.keys()):
    new_key = key
    if not key.startswith('transformer.'):
      new_key = f'transformer.{key}'
      output_state_dict[new_key] = output_state_dict.pop(key)
    print(f'name: {new_key}, shape:{output_state_dict[new_key].shape}')
  return config, megatron_args, iteration, output_state_dict

def convert_hf_to_megatron(
  config,
  hf_state_dict,
  output_megatron_path,
  target_megatron_tp,
  target_megatron_pp,
  margs,
  iteration,
  custom_layer_split=None):
  assert not (target_megatron_pp is not None and custom_layer_split is not None), \
    (target_megatron_pp, custom_layer_split)
    
  if custom_layer_split is not None:
    assert isinstance(custom_layer_split,list), custom_layer_split
    assert custom_layer_split, custom_layer_split
    
  if target_megatron_pp is not None:
    if config.num_hidden_layers % target_megatron_pp != 0:
      raise ValueError(
          f"Number of layers ({config.num_hidden_layers}) must be divisible by number of tensor parallelism"
          f" ({target_megatron_pp})")
    num_layers = config.num_hidden_layers // target_megatron_pp
    layer_splits = [num_layers for _ in range(target_megatron_pp)]
  else:
    # 保证layer的总和等于num_hedden_layers
    assert sum(custom_layer_split) == config.num_hidden_layers,custom_layer_split
    target_megatron_pp = len(custom_layer_split)
    layer_splits = custom_layer_split

  # for k,v in state_dict.items():
  #   if isinstance(v,dict):
  #     print(k,v)
  #   else:

  #     print(k,v.shape)

  # print(sdg)
  
  # Saving the tracker file
  os.makedirs(output_megatron_path, exist_ok=True)
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
  word_embedding = hf_state_dict["transformer.word_embeddings.weight"].to(dtype)
  orig_vocab_size = config.vocab_size
  padded_vocab_size = _vocab_size_with_padding(orig_vocab_size, margs)
  setattr(margs, "padded_vocab_size", padded_vocab_size)


  setattr(margs,'tensor_model_parallel_size',target_megatron_tp)
  setattr(margs,'pipeline_model_parallel_size',target_megatron_pp)

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
  word_emb_dict["norm.weight"] = hf_state_dict["transformer.word_embeddings_layernorm.weight"].to(dtype)
  word_emb_dict["norm.bias"] = hf_state_dict["transformer.word_embeddings_layernorm.bias"].to(dtype)

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
          layer_name for layer_name in hf_state_dict.keys()
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

        params = hf_state_dict[layer_name].to(dtype)

        # print('op_name/weight_or_bias',op_name,weight_or_bias)
        # handle layernorm
        if op_name  in ['input_layernorm','post_attention_layernorm']:
          layer_name = f"layers.{layer}.{op_name}.{weight_or_bias}"

        # handle attention K, V, Q weights
        elif op_name.startswith(
            "self_attention.query_key_value") and weight_or_bias == "weight":
          # transformers stores D X (3*D) but Megatron-LM expects (3*D) X D.
          # params = params.transpose(0, 1).contiguous()

          # params = transformers_to_megatron_fix_query_key_value_ordering(
          #     params,
          #     3.0,
          #     3,
          #     heads,
          #     hidden_size_per_head,
          # )
          layer_name = f"layers.{layer}.self_attention.query_key_value.{weight_or_bias}"

        # handle attention K, V, Q bias
        # self_attention.query_key_value.bias
        elif op_name.startswith(
            "self_attention.query_key_value") and weight_or_bias == "bias":
          # params = transformers_to_megatron_fix_query_key_value_ordering(
          #     params,
          #     3.0,
          #     3,
          #     heads,
          #     hidden_size_per_head,
          # )
          layer_name = f"layers.{layer}.self_attention.query_key_value.{weight_or_bias}"

        elif (op_name == "attention.query_key_value" or
              op_name == "self_attention.query_key_value") and (
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
        params = hf_state_dict[f"transformer.ln_f.{weight_or_bias}"].to(dtype)
        layer_name = f"final_layernorm.{weight_or_bias}"
        for i in range(target_megatron_tp):
          params_dict = get_element_from_dict_by_path(
              output_state_dict[i], "model.language_model.encoder")
          params_dict[layer_name] = params
      
      # 处理v_head
      v_head_state_dict = {}
      for k,v in hf_state_dict.items():
        if k.startswith('v_head.'):
          v_head_state_dict[k.replace('v_head.','')] = v
      
      if v_head_state_dict:
        output_state_dict[0]['model']['v_head'] = v_head_state_dict


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
      checkpoint_dir = (f"mp_rank_{tp_rank:02d}" if target_megatron_pp == 1 else
                        f"mp_rank_{tp_rank:02d}_{pp_rank:03d}")
      # if use_distributed_optimizer:
      #   checkpoint_name = "model_rng.pt"
      # else:
      checkpoint_name = "model_optim_rng.pt"
      #   output_state_dict[tp_rank]["optimizer"] = dummy_optim_state_dict[
      #       "optimizer"]
      checkpoint_dir = os.path.join(output_megatron_path, iter_folder, checkpoint_dir)
      os.makedirs(checkpoint_dir, exist_ok=True)
      checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
      # if args.print_checkpoint_structure:
      print(
          f"Checkpoint structure of model state dict shard belonging to TP rank {tp_rank} and PP rank"
          f" {pp_rank}:")
      recursive_print(None, output_state_dict[tp_rank])
      print('save to ',checkpoint_path)
      torch.save(output_state_dict[tp_rank], checkpoint_path)

def main():
  args = parse_arguments()
  config, megatron_args, iteration, hf_output_state_dict = convert_deepspeed_to_hf(args)
  convert_hf_to_megatron(
    config,
    hf_output_state_dict,
    args.output_folder,
    args.target_tp,
    args.target_pp,
    megatron_args,
    iteration,
    custom_layer_split=args.custom_layer_split
    )
  
if __name__=='__main__':
  main()
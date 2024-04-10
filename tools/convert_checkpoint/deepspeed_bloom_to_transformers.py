#!/usr/bin/env python

import os
import torch
import json
import sys
from pathlib import Path

# insert megatron's root dir into sys.path
root_repo_path = str(Path(__file__).resolve().parents[2])
if root_repo_path not in sys.path:
  sys.path.insert(0, root_repo_path)

# from deepspeed.checkpoint import DeepSpeedCheckpoint
from deepspeed_to_megatron_helper import _create_rank_checkpoint, parse_arguments, DeepSpeedCheckpoint

# the import was tested to work with this version
# https://github.com/huggingface/transformers/commit/0af901e83 if it diverges we may consider
# copying that version here instead
# from transformers.models.megatron_gpt2.convert_megatron_gpt2_checkpoint import convert_megatron_checkpoint
from convert_megatron_bloom_checkpoint import convert_megatron_checkpoint
from transformers import GPT2Config, BloomConfig


def main():

  # this first part comes mainly from deepspeed_to_megatron.main
  args = parse_arguments()
  print(
      f'Converting DeepSpeed checkpoint in {args.input_folder} to HF Transformers checkpoint in {args.output_folder}'
  )

  ds_checkpoint = DeepSpeedCheckpoint(args.input_folder, args.target_tp,
                                      args.target_pp)
  iteration = ds_checkpoint.get_iteration()
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
  output_state_dict = convert_megatron_checkpoint(args, input_state_dict,
                                                  config)

  basename = args.output_folder
  os.makedirs(basename, exist_ok=True)

  # Print the structure of converted state dict.
  #if args.print_checkpoint_structure:
  #    recursive_print(None, output_state_dict)

  # Store the config to file.
  output_config_file = os.path.join(basename, "config.json")
  output_config = config.to_dict()
  output_config["architectures"] = ["BloomForCausalLM"]
  output_config["model_type"] = "bloom"
  print(f'Saving config to "{output_config_file}"')
  with open(output_config_file, "w") as f:
    json.dump(output_config, f)

  # Store the state_dict to file.
  output_checkpoint_file = os.path.join(basename, "pytorch_model.bin")
  print(f'Saving checkpoint to "{output_checkpoint_file}"')
  torch.save(output_state_dict, output_checkpoint_file)

  print("Now add tokenizer files and upload to the hub")


if __name__ == "__main__":
  main()

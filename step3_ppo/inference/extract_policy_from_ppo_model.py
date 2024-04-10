"""
从actor_critic的checkpoint中提取出actor
"""
import torch
import argparse
import glob
import os
import re
from functools import lru_cache
from tools.convert_checkpoint.deepspeed_to_megatron_helper import \
  _create_checkpoint_paths,_create_latest_file,_save_checkpoint


def parse_arguments():
  parser = argparse.ArgumentParser()
  parser.add_argument('--input_folder',
                      default=None,
                      type=str,
                      help='actor-critic model input folder')

  parser.add_argument('--output_folder', type=str, help='output folder')

  parser.add_argument('--actor_num_layers',
                      type=int,
                      default=None,
                      help='actor num layers')

  parser.add_argument('--actor_hidden_size',
                      type=int,
                      default=None,
                      help='actor hidden size')

  parser.add_argument('--actor_num_attention_heads',
                      type=int,
                      default=None,
                      help='actor num attention heads')

  parser.add_argument('--iteration', type=int, default=0, help='iteration')
  args = parser.parse_args()
  print(f'args = {args}')
  return args


def find_megatron_rank_save_folder(input_folder):
  # 返回按照rank的目录
  rank_folders = glob.glob(input_folder + '/mp_rank_*')
  return rank_folders


@lru_cache()
def load_megatron_state_dict(rank_folder):
  return torch.load(rank_folder + "/model_optim_rng.pt")


def parse_tp_pp(rank_folder):
  basename = os.path.basename(rank_folder)
  groups = re.match('mp_rank_([0-9]+)[_]*([0-9]*)', basename).groups()
  tp = int(groups[0])
  pp = 0 if not groups[1] else int(groups[1])
  return tp, pp


def find_tp_pp(rank_folders):
  tps = []
  pps = []
  for rank_folder in rank_folders:
    tp, pp = parse_tp_pp(rank_folder)
    tps.append(tp)
    pps.append(pp)
  return max(tps) + 1, max(pps) + 1


def main():
  args = parse_arguments()
  actor_critic_input_folder = args.input_folder

  output_folder = args.output_folder
  #
  actor_rank_folders = find_megatron_rank_save_folder(
      actor_critic_input_folder)

  print(actor_rank_folders)

  # folder_no_diff = set([os.path.basename(i) for i in actor_rank_folders]) == \
  #   set([os.path.basename(i) for i in critic_rank_folders])
  # assert folder_no_diff, (actor_rank_folders,critic_rank_folders)

  actor_critic_tp, actor_critic_pp = find_tp_pp(actor_rank_folders)
  # critic_tp, critic_pp = find_tp_pp(critic_rank_folders)

  # assert actor_tp == critic_tp, (actor_tp,critic_tp)
  # assert actor_pp == critic_pp, (actor_pp,critic_pp)
  print(f'current tp: {actor_critic_tp}, pp: {actor_critic_pp}')
  _create_latest_file(output_folder, args.iteration)
  # 构建output路径
  checkpoint_paths = _create_checkpoint_paths(output_folder, args.iteration,
                                              actor_critic_tp, actor_critic_pp)
  for checkpoint_path in sum(checkpoint_paths, []):

    basename = os.path.basename(os.path.dirname(checkpoint_path))
    actor_critic_state_dict = load_megatron_state_dict(
        os.path.join(actor_critic_input_folder, basename))

    _args = actor_critic_state_dict['args']
    _args.num_layers = args.actor_num_layers or _args.num_layers
    _args.hidden_size = args.actor_hidden_size or _args.hidden_size
    _args.num_attention_heads = args.actor_num_attention_heads or _args.num_attention_heads

    # critic_state_dict = load_megatron_state_dict(os.path.join(critic_input_folder,basename))
    merge_model_state_dict = {
        "iteration": _args.iteration,
        'model': actor_critic_state_dict['model']['actor'],
        "checkpoint_version": actor_critic_state_dict['checkpoint_version'],
        'args': _args
    }
    # 保存
    print(f'saving to {checkpoint_path}')
    _save_checkpoint(checkpoint_path, merge_model_state_dict)
    del actor_critic_state_dict

    del merge_model_state_dict


if __name__ == "__main__":
  main()

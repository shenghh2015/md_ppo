from tools.convert_checkpoint.megatron_to_transformers import recursive_print
import argparse
import torch

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path')
    
    return parser.parse_args()

def main():

    args = get_args()
    model = torch.load(args.model_path)
    recursive_print('model', model)

if __name__ == '__main__':
    main()
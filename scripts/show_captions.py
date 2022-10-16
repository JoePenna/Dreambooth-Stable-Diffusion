#!/bin/env python3
import torch
import sys, os

def main(checkpoint_path):
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if 'trained_captions' in checkpoint:
            captions = checkpoint['trained_captions']
            global_step = checkpoint['global_step']
            print(f'Captions in {os.path.basename(checkpoint_path)} [{global_step} Global Steps]:')
            for caption in captions:
                print(f'\t"{caption}"')
        else:
            print(f'{checkpoint_path} has no captions saved')
    except:
        print(f'Failed to extract captions from {checkpoint_path}')

if __name__ == '__main__':
    if len(sys.argv) == 1:
        print(f'{sys.argv[0]} <checkpoint>')
    main(sys.argv[1])


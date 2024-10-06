import torch
import os
from os.path import join as pjoin
import codecs as cs
import numpy as np
import librosa
import random
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
import argparse
from dtw import *
from einops import rearrange

import sys
from pathlib import Path
# Get the directory of the current script
current_dir = Path(__file__).parent
# Get the parent directory
parent_dir = current_dir.parent
# Add the parent directory to sys.path
sys.path.append(str(parent_dir))

from unimumo.alignment import visual_beat, interpolation
from unimumo.motion import motion_process
from unimumo.util import load_model_from_config, interpolate_to_60fps


# randomly pair each music with several aligned motion


def main(args):
    # data paths and save paths
    motion_dir = 'data/motion'
    motion_feature_save_dir = 'data/motion/motion_code'
    # model paths
    ckpt = 'pretrained/motion_vqvae.ckpt'
    yaml_dir = 'configs/train_motion_vqvae.yaml'

    # set the fps of motion vqvae, should be 20 or 60
    fps = 60

    os.makedirs(motion_feature_save_dir, exist_ok=True)
    seed_everything(2023)

    # load motion vqvae model
    config = OmegaConf.load(yaml_dir)
    model = load_model_from_config(config, ckpt)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    model.eval()

    # load motion data
    ignore = []
    motion_data = {'train': [], 'test': [], 'val': []}
    aist = []
    dancedb = []
    humanml3d = []
    motion_mean = np.load(pjoin(motion_dir, 'Mean.npy'))
    motion_std = np.load(pjoin(motion_dir, 'Std.npy'))

    with cs.open(pjoin(motion_dir, 'ignore_list.txt'), "r") as f:
        for line in f.readlines():
            ignore.append(line.strip())
    for split in ['train', 'test', 'val']:
        with cs.open(pjoin(motion_dir, f'humanml3d_{split}.txt'), "r") as f:
            for line in f.readlines():
                if line.strip() in ignore:
                    continue
                if not os.path.exists(pjoin(motion_dir, split, 'joint_vecs', line.strip() + '.npy')):
                    continue
                motion_data[split].append(line.strip())
                humanml3d.append(line.strip())
        with cs.open(pjoin(motion_dir, f'aist_{split}.txt'), "r") as f:
            for line in f.readlines():
                if line.strip() in ignore:
                    continue
                if not os.path.exists(pjoin(motion_dir, split, 'joint_vecs', line.strip() + '.npy')):
                    continue
                motion_data[split].append(line.strip())
                aist.append(line.strip())
        with cs.open(pjoin(motion_dir, f'dancedb_{split}.txt'), "r") as f:
            for line in f.readlines():
                if line.strip() in ignore:
                    continue
                if not os.path.exists(pjoin(motion_dir, split, 'joint_vecs', line.strip() + '.npy')):
                    continue
                motion_data[split].append(line.strip())
                dancedb.append(line.strip())

    for split in ['train', 'test', 'val']:
        start_idx = int(args.start * len(motion_data[split]))
        end_idx = int(args.end * len(motion_data[split]))
        motion_data[split] = motion_data[split][start_idx:end_idx]

    num_motion = len(motion_data['train']) + len(motion_data['val']) + len(motion_data['test'])
    print(f'total motion: {num_motion}')
    print(f'length of dance: {len(dancedb) + len(aist)}')
    print(f'length of non dance: {len(humanml3d)}')

    for split in ['train', 'test', 'val']:
        for data_idx, motion_id in enumerate(motion_data[split]):
            if os.path.exists(pjoin(motion_feature_save_dir, motion_id + '.pth')):
                continue

            motion = np.load(pjoin(motion_dir, split, 'joint_vecs', motion_id + '.npy'))

            if motion_id in humanml3d or motion_id in dancedb:
                motion = interpolate_to_60fps(motion)

            # max length of 10
            print(f'{split}, {data_idx}, motion shape: {motion.shape} -> ', end='')
            max_motion_length = 10 * fps
            motion = motion[:max_motion_length]
            if motion.shape[0] < max_motion_length:
                # zero-pad to target length
                padded_motion = np.zeros((max_motion_length, motion.shape[-1]))
                padded_motion[:motion.shape[0]] = motion
                motion = padded_motion
            print(motion.shape)

            motion = (motion - motion_mean) / motion_std
            motion = torch.FloatTensor(motion)  # T, D
            motion = motion[None, ...]

            zero_waveform = torch.zeros((1, 1, 32000 * 10))

            music_emb, motion_emb = model.encode(zero_waveform.to(device), motion.to(device))
            motion_code = model.quantizer.encode(motion_emb)
            motion_token = motion_code.squeeze()

            print(f', code shape: {motion_token.shape}')
            motion_token = motion_token.cpu()

            motion_token_save_path = pjoin(motion_feature_save_dir, motion_id + '.pth')
            torch.save(motion_token, motion_token_save_path)  # 4, 1500


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-s',
        "--start",
        type=float,
        required=False,
        default=0.0,
        help='the start ratio for this preprocessing'
    )
    parser.add_argument(
        '-e',
        "--end",
        type=float,
        required=False,
        default=1.0,
        help='the end ratio of this preprocessing'
    )
    args = parser.parse_args()

    main(args)

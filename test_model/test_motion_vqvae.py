import random
from omegaconf import OmegaConf
import argparse
import os
import torch
from os.path import join as pjoin
import numpy as np
import codecs as cs
import librosa
import soundfile as sf
from pytorch_lightning import seed_everything
from einops import rearrange

import sys
from pathlib import Path
# Get the directory of the current script
current_dir = Path(__file__).parent
# Get the parent directory
parent_dir = current_dir.parent
# Add the parent directory to sys.path
sys.path.append(str(parent_dir))

from unimumo.util import load_model_from_config, interpolate_to_60fps
from unimumo.motion.motion_process import motion_vec_to_joint
from unimumo.motion import skel_animation
from unimumo.motion.utils import kinematic_chain
from unimumo.audio.audiocraft_.models.builders import get_compression_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--motion_dir",
        type=str,
        required=False,
        help="The path to motion data dir",
        default='data/motion'
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        required=False,
        help="The path to music data dir",
        default="exp_result/test_motion_vqvae_samples"
    )

    parser.add_argument(
        "--ckpt",
        type=str,
        required=False,
        default='pretrained/motion_vqvae.ckpt',
        help="load checkpoint",
    )

    parser.add_argument(
        "--encodec_ckpt",
        type=str,
        required=False,
        default='pretrained/music_vqvae.bin',
        help="load components from pretrained Encodec",
    )

    parser.add_argument(
        "--base",
        type=str,
        required=False,
        default='configs/train_motion_vqvae.yaml',
        help="yaml dir",
    )

    parser.add_argument(
        "--fps",
        type=int,
        required=False,
        default=60,
        choices=[20, 60],
        help="fps to load motion data",
    )

    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    seed_everything(2023)

    motion_dir = args.motion_dir
    fps = args.fps
    mean = np.load(pjoin(motion_dir, 'Mean.npy'))
    std = np.load(pjoin(motion_dir, 'Std.npy'))

    motion_ignore_list = []
    motion_data_list = []
    aist = []
    dancedb = []

    with cs.open(pjoin(motion_dir, 'ignore_list.txt'), "r") as f:
        for line in f.readlines():
            motion_ignore_list.append(line.strip())

    with cs.open(pjoin(motion_dir, f'aist_test.txt'), "r") as f:
        for line in f.readlines():
            if line.strip() in motion_ignore_list:
                continue
            if not os.path.exists(pjoin(motion_dir, 'test', 'joint_vecs', line.strip() + '.npy')):
                continue
            motion_data_list.append(line.strip())
            aist.append(line.strip())
    with cs.open(pjoin(motion_dir, f'dancedb_test.txt'), "r") as f:
        for line in f.readlines():
            if not os.path.exists(pjoin(motion_dir, 'test', 'joint_vecs', line.strip() + '.npy')):
                continue
            motion_data_list.append(line.strip())
            dancedb.append(line.strip())

    print('number of testing data:', len(motion_data_list))

    config = OmegaConf.load(args.base)
    # make sure that the provided or default path for music vqvae exists
    if not os.path.exists(config.model.params.music_config.vqvae_ckpt):
        assert os.path.exists(args.encodec_ckpt), (f'The default path in config does not exist: '
                                                   f'{config.model.params.music_config.vqvae_ckpt}.'
                                                   f'Please specify the correct path for pretrained Encodec')
        config.model.params.music_config.vqvae_ckpt = args.encodec_ckpt
    model = load_model_from_config(config, args.ckpt)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    model.eval()

    count = 0
    total_num = len(motion_data_list)
    loss = 0
    loss_each_sample = []
    while count < total_num:
        motion_name = motion_data_list[count]
        motion = np.load(pjoin(motion_dir, 'test', 'joint_vecs', motion_name + '.npy'))

        if motion_name in aist and fps == 20:
            motion = motion[::3]
        if motion_name not in aist and fps == 60:
            motion = interpolate_to_60fps(motion)

        # pad motion into integer length
        motion_length = motion.shape[0]
        duration = motion_length // fps + 1

        motion = (motion - mean) / std
        motion = torch.tensor(motion)

        padded_motion = torch.zeros((duration * fps, 263))
        padded_motion[:motion_length] = motion
        motion = padded_motion[None, ...]

        music_target_length = int(duration * 32000)
        waveform = torch.zeros((1, 1, music_target_length))

        print(f'music shape: {waveform.shape}, motion shape: {motion.shape}')

        with torch.no_grad():
            motion = motion.to(device)
            waveform = waveform.to(device)

            batch = {
                'motion': motion,
                'music': waveform
            }

            motion_recon = model.forward(batch)[0]

            curr_loss = torch.nn.functional.mse_loss(motion, motion_recon)
            print(f'{count + 1}/{total_num}, current loss: {curr_loss},', end=' ')
            if motion_name in aist:
                print('In AIST')
            elif motion_name in dancedb:
                print('In DanceDB')
            loss += curr_loss
            loss_each_sample.append(curr_loss.item())

        if count % 20 == 0:
            joint = motion_vec_to_joint(motion_recon, mean, std)
            gt_joint = motion_vec_to_joint(motion, mean, std)

            os.makedirs(args.save_dir, exist_ok=True)

            motion_filename = f'{count}_motion_recon.mp4'
            motion_save_path = pjoin(args.save_dir, motion_filename)
            skel_animation.plot_3d_motion(
                motion_save_path, kinematic_chain, joint[0], title='None', vbeat=None,
                fps=fps, radius=4
            )

            gt_motion_filename = f'{count}_motion_gt.mp4'
            motion_save_path = pjoin(args.save_dir, gt_motion_filename)
            skel_animation.plot_3d_motion(
                motion_save_path, kinematic_chain, gt_joint[0], title='None', vbeat=None,
                fps=fps, radius=4
            )

        count += 1

    total_loss = loss / total_num
    print(f'total loss: {total_loss}')

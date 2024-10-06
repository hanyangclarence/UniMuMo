import argparse
import json
import os

import numpy as np
import torch
from os.path import join as pjoin
import soundfile as sf
import pandas as pd
import subprocess
import random
from pytorch_lightning import seed_everything

import sys
from pathlib import Path
# Get the directory of the current script
current_dir = Path(__file__).parent
# Get the parent directory
parent_dir = current_dir.parent
# Add the parent directory to sys.path
sys.path.append(str(parent_dir))

from unimumo.motion import skel_animation
from unimumo.motion.utils import kinematic_chain
from unimumo.models import UniMuMo
from unimumo.util import interpolate_to_60fps

'''
Load paired music and motion, all made to 10 seconds
'''


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-s",
        "--save_path",
        type=str,
        required=False,
        help="The path to save model output",
        default="./test_motion2text_humanml3d",
    )

    parser.add_argument(
        "--motion_dir",
        type=str,
        required=False,
        help="The path to motion data dir",
        default='data/motion',
    )

    parser.add_argument(
        "--motion_code_dir",
        type=str,
        required=False,
        help="The path to motion data dir",
        default='data/motion/aligned_humanml3d_test_motion_code',
    )

    parser.add_argument(
        "-d",
        "--duration",
        type=float,
        required=False,
        default=10,
        help="Generated audio time",
    )

    parser.add_argument(
        "--seed",
        type=int,
        required=False,
        default=42,
        help="Change this value (any integer number) will lead to a different generation result.",
    )

    parser.add_argument(
        "--ckpt",
        type=str,
        required=False,
        default=None,
        help="load checkpoint",
    )

    parser.add_argument(
        "--start",
        type=float,
        required=False,
        default=0.,
        help="start ratio",
    )

    parser.add_argument(
        "--end",
        type=float,
        required=False,
        default=1.,
        help="end ratio",
    )

    args = parser.parse_args()

    seed_everything(args.seed)
    save_path = args.save_path
    motion_save_path = pjoin(save_path, 'motion')
    feature_263_save_path = pjoin(save_path, 'feature_263')
    feature_22_3_save_path = pjoin(save_path, 'feature_22_3')
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(motion_save_path, exist_ok=True)
    os.makedirs(feature_263_save_path, exist_ok=True)
    os.makedirs(feature_22_3_save_path, exist_ok=True)
    batch_size = 1
    motion_dir = args.motion_dir
    motion_code_dir = args.motion_code_dir
    duration = args.duration

    motion_id_list = []
    with open(pjoin(motion_dir, 'humanml3d_test.txt'), 'r') as f:
        for line in f.readlines():
            if os.path.exists(pjoin(motion_dir, 'test', 'joint_vecs', line.strip() + '.npy')):
                motion_id_list.append(line.strip())

    paired_music_motion = os.listdir(motion_code_dir)

    print('number of motion data:', len(motion_id_list), file=sys.stderr)
    print('number of paired motion: ', len(paired_music_motion), file=sys.stderr)

    # load model
    model = UniMuMo.from_checkpoint(args.ckpt)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    result_dict = {}
    count = 0
    with open(pjoin(save_path, f'gen_captions.json'), 'w') as f:
        while count < len(motion_id_list):
            print(f'{count}/{len(motion_id_list)}')
            motion_code_list = []
            motion_id = motion_id_list[count]

            motion_feature = np.load(pjoin(motion_dir, 'test', 'joint_vecs', motion_id + '.npy'))
            motion_feature = interpolate_to_60fps(motion_feature)
            print(f'motion feature: {motion_feature.shape} ->', end='')

            # repeat to the same size
            target_length = duration * model.motion_fps
            curr_lenth = motion_feature.shape[0]
            if curr_lenth >= target_length:
                motion_feature = motion_feature[:target_length]
            else:
                padded_feature = np.zeros((target_length, motion_feature.shape[-1]))
                padded_feature[:curr_lenth] = motion_feature
                motion_feature = padded_feature
            print(motion_feature.shape)

            captions = model.generate_text(motion_feature=motion_feature)

            # save some batch
            if count % 10 == 0:
                motion_id_to_save = motion_id
                motion_feature_to_save = np.load(pjoin(motion_dir, 'test', 'joint_vecs', motion_id_to_save + '.npy'))[None, ...]
                joint_to_save = model.motion_feature_to_joint(motion_feature_to_save)[0]

                motion_filename = "%s.mp4" % motion_id
                motion_path = pjoin(motion_save_path, motion_filename)
                try:
                    skel_animation.plot_3d_motion(
                        motion_path, kinematic_chain, joint_to_save, title='None', vbeat=None,
                        fps=model.motion_fps, radius=4
                    )
                except Exception as e:
                    print(e)

                feature_263_filename = "%s.npy" % motion_id
                feature_263_path = pjoin(feature_263_save_path, feature_263_filename)
                np.save(feature_263_path, motion_feature_to_save[0])

                feature_22_3_filename = "%s.npy" % motion_id
                feature_22_3_path = pjoin(feature_22_3_save_path, feature_22_3_filename)
                np.save(feature_22_3_path, joint_to_save)

            # write generated descriptions
            description = captions[0]
            description = description.replace('The motion is that', '')
            description = description.replace('The dance is that', '')
            description = description.strip().capitalize()

            result_dict[motion_id] = description
            print(f'\t\t\t\t{motion_id}\t{description}', file=sys.stderr)

            count += batch_size

        json.dump(result_dict, f, indent=4)
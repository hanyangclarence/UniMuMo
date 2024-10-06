import argparse
import os
import torch
from os.path import join as pjoin
import soundfile as sf
import numpy as np
import subprocess
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


# Test motion-to-music on AIST++ dataset.
# The data and split are downloaded from https://github.com/L-YeZhu/D2M-GAN (since I cannot find elsewhere
# the aligned music and motion of AIST++)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-s",
        "--save_path",
        type=str,
        required=False,
        help="The path to save model output",
        default="./test_motion2music_aist",
    )

    parser.add_argument(
        "--aist_dir",
        type=str,
        required=False,
        help="The path to music data dir",
        default="/gpfs/u/home/LMCG/LMCGnngn/scratch/yanghan/aist_plusplus_final"
    )

    parser.add_argument(
        "--motion_dir",
        type=str,
        required=False,
        help="The path to motion data dir",
        default='data/motion',
    )

    parser.add_argument(
        "-gs",
        "--guidance_scale",
        type=float,
        required=False,
        default=3.0,
        help="Guidance scale (Large => better quality and relavancy to text; Small => better diversity)",
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

    args = parser.parse_args()

    seed_everything(args.seed)
    save_path = args.save_path
    music_save_path = pjoin(save_path, 'music')
    motion_save_path = pjoin(save_path, 'motion')
    video_save_path = pjoin(save_path, 'video')
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(music_save_path, exist_ok=True)
    os.makedirs(motion_save_path, exist_ok=True)
    os.makedirs(video_save_path, exist_ok=True)
    guidance_scale = args.guidance_scale
    aist_dir = args.aist_dir
    motion_dir = args.motion_dir

    # read all aist motion data
    motion_data = {}
    for split in ['train', 'val', 'test']:
        data_list = os.listdir(pjoin(motion_dir, split, 'joint_vecs'))
        data_list = [s.split('.')[0] for s in data_list]
        data_list = [s for s in data_list if s[0] == 'g']
        motion_data[split] = data_list
        print(data_list[:20])

    motion_id_list = []
    with open(pjoin(aist_dir, 'aist_motion_test_segment.txt'), 'r') as f:
        for line in f.readlines():
            motion_id_list.append(line.strip())

    total_num = len(motion_id_list)
    print(f'total number of test data: {total_num}')
    print('train: ', len(motion_data['train']), 'test: ', len(motion_data['test']), 'val: ', len(motion_data['val']))

    # load model
    model = UniMuMo.from_checkpoint(args.ckpt)

    count = 0
    fps = model.motion_fps
    while count < total_num:
        # The data format in the directly downloaded aist++ is different from ours,
        # we just use their split and filename and load our data
        # And since their filename is slightly difference from our data
        # we need to change it name a bit...
        motion_id = motion_id_list[count]  # motion_s2/test/motion_id_segn.npy
        motion_id = motion_id.split('/')[-1].split('.')[0]
        print(f'{motion_id} -> ', end='')
        motion_name = '_'.join(motion_id.split('_')[:-1])
        motion_name = '_'.join(motion_name.split('_')[:2]) + '_cAll_' + '_'.join(motion_name.split('_')[-3:])
        seg_num = int(motion_id.split('_')[-1][3:])
        print(f'{motion_name}, segment {seg_num}, ', end='')

        if motion_name in motion_data['train']:
            motion_path = pjoin(motion_dir, 'train', 'joint_vecs', motion_name + '.npy')
            print('train')
        elif motion_name in motion_data['val']:
            motion_path = pjoin(motion_dir, 'val', 'joint_vecs', motion_name + '.npy')
            print('val')
        elif motion_name in motion_data['test']:
            motion_path = pjoin(motion_dir, 'test', 'joint_vecs', motion_name + '.npy')
            print('test')
        else:
            motion_path = None
        assert os.path.exists(motion_path)
        motion = np.load(motion_path)

        if fps == 20:
            # motion is in aist, so down sample by 3
            motion = motion[::3]
        # go to the specific segment
        motion = motion[(seg_num - 1) * fps * 2: seg_num * fps * 2]
        motion = motion[None, ...]
        print(f'motion shape: {motion.shape}')

        waveform_gen = model.generate_music_from_motion(
            motion_feature=motion,
            # text_description=['<music_prompt_start> The music is a rock, pop music, with fast tempo, which is intense. <music_prompt_end> '
            #                   '<motion_prompt_start> The genre of the dance is hip-hop. <motion_prompt_end>'],
            conditional_guidance_scale=guidance_scale
        )
        joint_gen = model.motion_vec_to_joint(
            torch.Tensor(model.normalize_motion(motion))
        )

        waveform_gen = waveform_gen.squeeze()
        joint_gen = joint_gen.squeeze()
        print(f'generate waveform: {waveform_gen.shape}, joint: {joint_gen.shape}')

        music_filename = "%s.mp3" % motion_id
        music_path = os.path.join(music_save_path, music_filename)
        try:
            sf.write(music_path, waveform_gen, 32000)
        except Exception as e:
            print(e)
            count += 1
            continue

        motion_filename = "%s.mp4" % motion_id
        motion_path = pjoin(motion_save_path, motion_filename)
        try:
            skel_animation.plot_3d_motion(
                motion_path, kinematic_chain, joint_gen, title='None', vbeat=None,
                fps=fps, radius=4
            )
        except Exception as e:
            print(e)
            count += 1
            continue

        video_filename = "%s.mp4" % motion_id
        video_path = pjoin(video_save_path, video_filename)
        try:
            subprocess.call(
                f"ffmpeg -i {motion_path} -i {music_path} -c copy {video_path}",
                shell=True)
        except Exception as e:
            print(e)
            count += 1
            continue

        count += 1

import argparse
import os
import torch
from os.path import join as pjoin
import soundfile as sf
import numpy as np
import codecs as cs
import random
import json
from pytorch_lightning import seed_everything
import subprocess

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-s",
        "--save_path",
        type=str,
        required=False,
        help="The path to save model output",
        default="./demo_motiontext2music",
    )

    parser.add_argument(
        "--music_meta_dir",
        type=str,
        required=False,
        help="The path to meta data dir",
        default="data/music",
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
    feature_263_save_path = pjoin(save_path, 'feature_263')
    feature_22_3_save_path = pjoin(save_path, 'feature_22_3')
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(music_save_path, exist_ok=True)
    os.makedirs(motion_save_path, exist_ok=True)
    os.makedirs(video_save_path, exist_ok=True)
    os.makedirs(feature_263_save_path, exist_ok=True)
    os.makedirs(feature_22_3_save_path, exist_ok=True)
    guidance_scale = args.guidance_scale
    music_meta_dir = args.music_meta_dir
    motion_dir = args.motion_dir
    duration = 10

    motion_id_list = []
    with open(pjoin(motion_dir, 'aist_test.txt'), 'r') as f:
        for line in f.readlines():
            motion_id_list.append(line.strip())

    aist_genre_map = {
        'gBR': 'break',
        'gPO': 'pop',
        'gLO': 'lock',
        'gMH': 'middle hip-hop',
        'gLH': 'LA style hip-hop',
        'gHO': 'house',
        'gWA': 'waack',
        'gKR': 'krump',
        'gJS': 'street jazz',
        'gJB': 'ballet jazz'
    }

    motion_description_list = []
    for motion_id in motion_id_list:
        genre_id = motion_id.split('_')[0]
        genre = aist_genre_map[genre_id]
        desc_choices = [f'The genre of the dance is {genre}.', f'The style of the dance is {genre}.',
                        f'This is a {genre} style dance.']
        dance_description = random.choice(desc_choices)
        motion_description_list.append(dance_description)

    with open(pjoin(music_meta_dir, 'music4all_captions_mullama.json'), 'r') as caption_fd:
        caption_dict = json.load(caption_fd)
        music_description_list = [caption_dict[k] for k in caption_dict.keys()]
        music_description_list = [s for s in music_description_list if 'male vocalist' not in s]

    text_prompt_list = []
    for i, motion_description in enumerate(motion_description_list):
        music_description = random.choice(music_description_list)
        text_prompt = music_description + ' <separation> ' + motion_description
        text_prompt_list.append(text_prompt)

    assert len(text_prompt_list) == len(motion_id_list)

    with open(pjoin(save_path, 'text_prompt.txt'), 'w') as f:
        for i, text_prompt in enumerate(text_prompt_list):
            f.write(motion_id_list[i] + '\t' + text_prompt + '\n')

    # load model
    model = UniMuMo.from_checkpoint(args.ckpt)

    count = 0
    total_num = len(motion_id_list)
    while count < total_num:
        motion_id = motion_id_list[count]
        motion_path = pjoin(motion_dir, 'test', 'joint_vecs', motion_id + '.npy')

        if not os.path.exists(motion_path):
            print(f'{motion_path} does not exist!')
            count += 1
            continue

        motion = np.load(motion_path)

        target_length = duration * model.motion_fps
        target_length = min(target_length, motion.shape[0])
        target_length = (target_length // model.motion_fps) * model.motion_fps
        start_idx = random.randint(0, motion.shape[0] - target_length)
        motion = motion[start_idx:start_idx + target_length]
        motion = motion[None, ...]  # (1, fps * duration, D)

        waveform_gen = model.generate_music_from_motion(
            motion_feature=motion,
            text_description=[text_prompt_list[count]],
            conditional_guidance_scale=guidance_scale,
        )

        joint_gen = model.motion_vec_to_joint(
            torch.Tensor(model.normalize_motion(motion))
        )
        print(f'waveform gen: {waveform_gen.shape}, joint gen: {joint_gen.shape}, motion shape: {motion.shape}')

        os.makedirs(save_path, exist_ok=True)

        music_filename = "%s.mp3" % motion_id
        music_path = os.path.join(music_save_path, music_filename)
        try:
            sf.write(music_path, waveform_gen.squeeze(), 32000)
        except Exception as e:
            print(e)
            count += 1
            continue

        motion_filename = "%s.mp4" % motion_id
        motion_path = pjoin(motion_save_path, motion_filename)
        try:
            skel_animation.plot_3d_motion(
                motion_path, kinematic_chain, joint_gen.squeeze(), title='None', vbeat=None,
                fps=model.motion_fps, radius=4
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

        feature_263_filename = "%s.npy" % motion_id
        feature_263_path = pjoin(feature_263_save_path, feature_263_filename)
        np.save(feature_263_path, motion.squeeze())

        feature_22_3_filename = "%s.npy" % motion_id
        feature_22_3_path = pjoin(feature_22_3_save_path, feature_22_3_filename)
        np.save(feature_22_3_path, joint_gen.squeeze())

        count += 1

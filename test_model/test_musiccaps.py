import argparse
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-s",
        "--save_path",
        type=str,
        required=False,
        help="The path to save model output",
        default="./test_result_on_musiccap",
    )

    parser.add_argument(
        "--musiccaps_dir",
        type=str,
        required=False,
        help="The path to the directory containing musiccaps prompts",
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

    parser.add_argument(
        "--batch_size",
        type=int,
        required=False,
        default=20,
        help="batch size for inference",
    )

    parser.add_argument(
        "--aist_prob",
        type=float,
        required=False,
        default=0.8,
        help="Prob of choosing AIST style motion caption.",
    )

    parser.add_argument(
        "--save_wav_only",
        type=bool,
        required=False,
        default=True,
        help="Only save waveform",
    )

    args = parser.parse_args()

    seed_everything(args.seed)
    save_path = args.save_path
    music_save_path = pjoin(save_path, 'music')
    motion_save_path = pjoin(save_path, 'motion')
    video_save_path = pjoin(save_path, 'video')
    joint_save_path = pjoin(save_path, 'joint')
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(music_save_path, exist_ok=True)
    os.makedirs(motion_save_path, exist_ok=True)
    os.makedirs(video_save_path, exist_ok=True)
    os.makedirs(joint_save_path, exist_ok=True)
    guidance_scale = args.guidance_scale
    motion_dir = args.motion_dir
    duration = args.duration

    # load random motion descriptions
    humanml3d_text_dir = pjoin(motion_dir, 'humanml3d_text_description')
    descriptions = os.listdir(humanml3d_text_dir)[:500]
    humanml3d_text = []
    for desc_txt in descriptions:
        with open(pjoin(motion_dir, 'humanml3d_text_description', desc_txt), 'r', encoding='UTF-8') as f:
            texts = []
            lines = f.readlines()
            for line in lines:
                text = line.split('#')[0]
                if text[-1] == '.':
                    text = text[:-1]
                humanml3d_text.append(text)
    print(f'Loaded {len(humanml3d_text)} text prompts from humanml3d')

    aist_genres = ['break', 'pop', 'lock', 'middle hip-hop', 'house', 'waack', 'krump', 'street jazz', 'ballet jazz']

    # load musiccaps text prompt
    assert os.path.exists(args.musiccaps_dir)
    music_cap_df = pd.read_csv(pjoin(args.musiccaps_dir, 'musiccaps-public.csv'))
    text_prompt_list = list(music_cap_df['caption'])
    music_id_list = list(music_cap_df['ytid'])

    print('number of testing data:', len(text_prompt_list))

    # load model
    model = UniMuMo.from_checkpoint(args.ckpt)

    count = 0
    total_num = len(text_prompt_list)
    start_idx = int(args.start * total_num)
    end_idx = int(args.end * total_num)
    count = max(start_idx, count)
    print(f'start: {count}, end: {end_idx}')
    while count < end_idx:
        # text condition
        text_prompt_full = text_prompt_list[count:min(end_idx, count + args.batch_size)]
        music_id_full = music_id_list[count:min(end_idx, count + args.batch_size)]
        print(f'{count + 1}-{min(end_idx, count + args.batch_size)}/{total_num}', end=', ')

        # check whether each file has existed
        text_prompt = []
        music_id = []
        for batch_idx in range(len(text_prompt_full)):
            if os.path.exists(pjoin(music_save_path, f'{music_id_full[batch_idx]}.mp3')):
                continue
            else:
                music_description = text_prompt_full[batch_idx]
                motion_description = None
                if random.uniform(0, 1) < args.aist_prob:
                    # use aist style prompts
                    genre = random.choice(aist_genres)
                    motion_description = f'The style of the dance is {genre}.'
                else:
                    motion_description = random.choice(humanml3d_text)
                music_motion_description = music_description + ' <separation> ' + motion_description
                text_prompt.append(music_motion_description)
                music_id.append(music_id_full[batch_idx])

        if len(text_prompt) == 0:
            print(f'{count}-{count + args.batch_size} exists!')
            count += args.batch_size
            continue

        print(f'generating {len(text_prompt)} audio')

        for p in text_prompt:
            print(len(p.split(' ')), p)

        with torch.no_grad():
            waveform_gen, motion_gen = model.generate_music_motion(
                text_description=text_prompt,
                duration=duration,
                conditional_guidance_scale=guidance_scale
            )

            os.makedirs(save_path, exist_ok=True)

            for batch_idx in range(len(text_prompt)):

                if args.save_wav_only:
                    music_filename = "%s.wav" % music_id[batch_idx]
                    music_path = os.path.join(music_save_path, music_filename)
                    try:
                        sf.write(music_path, waveform_gen[batch_idx], 32000)
                    except Exception as e:
                        print(e)
                        continue
                else:
                    music_filename = "%s.mp3" % music_id[batch_idx]
                    music_path = os.path.join(music_save_path, music_filename)
                    try:
                        sf.write(music_path, waveform_gen[batch_idx], 32000)
                    except Exception as e:
                        print(e)
                        continue

                    motion_filename = "%s.mp4" % music_id[batch_idx]
                    motion_path = pjoin(motion_save_path, motion_filename)
                    try:
                        skel_animation.plot_3d_motion(
                            motion_path, kinematic_chain, motion_gen['joint'][batch_idx], title='None', vbeat=None,
                            fps=model.motion_fps, radius=4
                        )
                    except Exception as e:
                        print(e)
                        continue

                    video_filename = "%s.mp4" % music_id[batch_idx]
                    video_path = pjoin(video_save_path, video_filename)
                    try:
                        subprocess.call(f"ffmpeg -i {motion_path} -i {music_path} -c copy {video_path}", shell=True)
                    except Exception as e:
                        print(e)
                        continue

                    joint_filename = "%s.npy" % music_id[batch_idx]
                    joint_path = pjoin(joint_save_path, joint_filename)
                    np.save(joint_path, motion_gen['joint'][batch_idx])

        count += args.batch_size

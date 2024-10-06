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
import librosa

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
        default="./demo_musictext2motion",
    )

    parser.add_argument(
        "--audio_dir",
        type=str,
        required=False,
        help="The path to audio file dir",
        default="data/music/audios"
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

    parser.add_argument(
        "--batch_size",
        type=int,
        required=False,
        default=20,
        help="batch size for inference",
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
    duration = 10
    audio_dir = args.audio_dir
    music_meta_dir = args.music_meta_dir

    music_ignore_list = []
    music_data_list = []

    assert os.path.exists(music_meta_dir)

    with cs.open(pjoin(music_meta_dir, f'music4all_ignore.txt'), 'r') as f:
        for line in f.readlines():
            music_ignore_list.append(line.strip())

    with cs.open(pjoin(music_meta_dir, f'music4all_test.txt'), "r") as f:
        for line in f.readlines():
            if line.strip() in music_ignore_list:
                continue
            music_data_list.append(line.strip())
    print('number of testing data:', len(music_data_list))

    aist_genres = ['break', 'pop', 'lock', 'middle hip-hop', 'LA style hip-hop', 'house', 'waack', 'krump', 'street jazz', 'ballet jazz']

    # Some text prompt
    text_prompt_list = []
    music_id_list = []

    music_prompt_list = []
    with open(pjoin(music_meta_dir, 'music4all_captions_mullama.json'), 'r') as caption_fd:
        music_caption = json.load(caption_fd)

    for music_id in music_data_list:
        if music_id not in music_caption.keys():
            continue
        music_id_list.append(music_id)
        music_prompt_list.append(music_caption[music_id])

    for i in range(len(music_prompt_list)):
        genre = random.choice(aist_genres)
        desc_choices = [f'The genre of the dance is {genre}.', f'The style of the dance is {genre}.',
                        f'This is a {genre} style dance.']
        dance_description = random.choice(desc_choices)
        full_description = music_prompt_list[i] + ' <separation> ' + dance_description
        text_prompt_list.append(full_description)

    with cs.open(pjoin(save_path, 'text_prompt.txt'), 'w') as f:
        for i, text_prompt in enumerate(text_prompt_list):
            f.write(music_id_list[i] + '\t' + text_prompt + '\n')

    # load model
    model = UniMuMo.from_checkpoint(args.ckpt)

    total_num = len(music_id_list)
    print(f'total number of test data: {total_num}')
    count = 0
    while count < total_num:
        text_prompt_full = text_prompt_list[count:count + args.batch_size]
        music_id_full = music_id_list[count:count + args.batch_size]
        print(f'{count + 1}-{min(total_num, count + args.batch_size)}/{total_num}', end=', ')

        batch_waveform = []
        target_length = duration * 32000
        for music_id in music_id_full:
            music_path = pjoin(audio_dir, music_id + '.wav')
            waveform, _ = librosa.load(music_path, sr=32000)
            waveform = waveform[:target_length]
            waveform = waveform[None, None, ...]
            batch_waveform.append(waveform)

        batch_waveform = np.concatenate(batch_waveform, axis=0)

        motion_gen = model.generate_motion_from_music(
            waveform=batch_waveform,
            text_description=text_prompt_full,
            conditional_guidance_scale=guidance_scale,
        )
        print(f"input waveform: {batch_waveform.shape}, joint: {motion_gen['joint'].shape}, feature: {motion_gen['feature'].shape}")

        os.makedirs(save_path, exist_ok=True)

        for batch_idx in range(len(text_prompt_full)):
            music_filename = "%s.mp3" % music_id_full[batch_idx]
            music_path = os.path.join(music_save_path, music_filename)
            try:
                sf.write(music_path, batch_waveform[batch_idx].squeeze(), 32000)
            except Exception as e:
                print(e)
                continue

            motion_filename = "%s.mp4" % music_id_full[batch_idx]
            motion_path = pjoin(motion_save_path, motion_filename)
            try:
                skel_animation.plot_3d_motion(
                    motion_path, kinematic_chain, motion_gen['joint'][batch_idx], title='None', vbeat=None,
                    fps=model.motion_fps, radius=4
                )
            except Exception as e:
                print(e)
                continue

            video_filename = "%s.mp4" % music_id_full[batch_idx]
            video_path = pjoin(video_save_path, video_filename)
            try:
                subprocess.call(
                    f"ffmpeg -i {motion_path} -i {music_path} -c copy {video_path}",
                    shell=True)
            except Exception as e:
                print(f'{video_path} cannot be saved.')
                continue

            feature_263_filename = "%s.npy" % music_id_full[batch_idx]
            feature_263_path = pjoin(feature_263_save_path, feature_263_filename)
            np.save(feature_263_path, motion_gen['feature'][batch_idx])

            feature_22_3_filename = "%s.npy" % music_id_full[batch_idx]
            feature_22_3_path = pjoin(feature_22_3_save_path, feature_22_3_filename)
            np.save(feature_22_3_path, motion_gen['joint'][batch_idx])

        count += args.batch_size

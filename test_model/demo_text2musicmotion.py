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
        default="./demo_text2musicmotion",
    )

    parser.add_argument(
        "--music_meta_dir",
        type=str,
        required=False,
        help="The path to meta data dir",
        default="data/music",
    )

    parser.add_argument(
        "-gs",
        "--guidance_scale",
        type=float,
        required=False,
        default=4,
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
    duration = args.duration

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

    # load text prompt
    aist_genres = ['break', 'pop', 'lock', 'middle hip-hop', 'LA style hip-hop', 'house', 'waack', 'krump', 'street jazz', 'ballet jazz']
    text_prompt_list = []
    music_id_list = []
    music_prompt_list = []

    with open(pjoin(music_meta_dir, 'music4all_captions_mullama.json'), 'r') as caption_fd:
        mullama_caption = json.load(caption_fd)
    with open(pjoin(music_meta_dir, 'music4all_captions_gpt.json'), 'r') as caption_fd:
        gpt_caption = json.load(caption_fd)

    for music_id in music_data_list:
        current_caption = []
        if music_id in mullama_caption.keys():
            if 'male vocalist' not in mullama_caption[music_id]:
                # as there are too much repetitive captions that contain "male vocalist" or "female vocalist",
                # we just remove these captions
                current_caption.append(mullama_caption[music_id])
        if music_id in gpt_caption.keys():
            current_caption.extend(gpt_caption[music_id])

        if len(current_caption) != 0:
            music_id_list.append(music_id)
            music_prompt_list.append(random.choice(current_caption))

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
                text_prompt.append(text_prompt_full[batch_idx])
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

            print(f"joint_gen: {motion_gen['joint'].shape}, waveform_gen: {waveform_gen.shape}")

            os.makedirs(save_path, exist_ok=True)

            for batch_idx in range(len(text_prompt)):
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
                    subprocess.call(
                        f"ffmpeg -i {motion_path} -i {music_path} -c copy {video_path}",
                        shell=True)
                except Exception as e:
                    print(e)
                    continue

                feature_263_filename = "%s.npy" % music_id[batch_idx]
                feature_263_path = pjoin(feature_263_save_path, feature_263_filename)
                np.save(feature_263_path, motion_gen['feature'][batch_idx])

                feature_22_3_filename = "%s.npy" % music_id[batch_idx]
                feature_22_3_path = pjoin(feature_22_3_save_path, feature_22_3_filename)
                np.save(feature_22_3_path, motion_gen['joint'][batch_idx])

        count += args.batch_size

    with cs.open(pjoin(save_path, 'text_prompt.txt'), 'w') as f:
        for i in range(total_num):
            music_id = music_id_list[i]
            if os.path.exists(pjoin(music_save_path, f'{music_id}.mp3')):
                f.write(f'{music_id}\t{text_prompt_list[i]}\n')

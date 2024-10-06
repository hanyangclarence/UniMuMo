import random
import librosa
import torch
from os.path import join as pjoin
import argparse
import numpy as np
import os
from pytorch_lightning import seed_everything
import json
import soundfile as sf
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


# Test the model on MusicQA dataset from MU-LLaMa: https://huggingface.co/datasets/mu-llama/MusicQA
# We select the questions like "Describe the audio" in the dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-s",
        "--save_path",
        type=str,
        required=False,
        help="The path to save model output",
        default="./test_mu2text_mullama",
    )

    parser.add_argument(
        "--test_music_dir",
        type=str,
        required=False,
        help="The path to the folder containing MusicQA audios",
        default='/gpfs/u/home/LMCG/LMCGnngn/scratch/yanghan/MU-LLaMA/data/audios'
    )

    parser.add_argument(
        "--mullama_meta_dir",
        type=str,
        required=False,
        help="The path to the directory that contains EvalMusicQA.txt",
        default="/gpfs/u/home/LMCG/LMCGnngn/scratch/yanghan/MU-LLaMA/data",
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
        "--visualize_result",
        type=bool,
        required=False,
        default=False,
        help="Whether to visualize music and motion",
    )

    parser.add_argument(
        "--start",
        type=float,
        required=False,
        default=0.0,
        help='the start ratio for this preprocessing'
    )

    parser.add_argument(
        "--end",
        type=float,
        required=False,
        default=1.0,
        help='the end ratio of this preprocessing'
    )

    args = parser.parse_args()

    seed_everything(args.seed)
    save_path = args.save_path
    music_save_path = pjoin(save_path, 'music')
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(music_save_path, exist_ok=True)
    mullama_meta_dir = args.mullama_meta_dir
    test_music_dir = args.test_music_dir
    visualize_result = args.visualize_result

    music_id_list = []
    music_description_list = []
    with open(pjoin(mullama_meta_dir, 'EvalMusicQA.txt'), 'r') as f:
        data_list = json.load(f)
        for data in data_list:
            if data['conversation'][0]['value'] == 'Describe the audio':
                description = data['conversation'][1]['value']
                if description == 'Describe the audio' or description == 'What is the audio about?' or description == 'What is the genre of the audio?':
                    continue
                if 'Describe the audio: ' in description:
                    description = description[len('Describe the audio: '):]

                music_id_list.append(data['audio_name'])
                music_description_list.append(data['conversation'][1]['value'])


    # load model
    model = UniMuMo.from_checkpoint(args.ckpt)

    total_num = len(music_id_list)
    print(f'total number of test data: {total_num}', file=sys.stderr)
    count = 0

    f_gen = open(pjoin(save_path, 'gen_captions.json'), 'w')
    f_gt = open(pjoin(save_path, 'gt_captions.json'), 'w')
    pred_caption = {}
    gt_caption = {}

    while count < total_num:
        music_id = music_id_list[count]
        waveform, sr = librosa.load(pjoin(test_music_dir, music_id), sr=32000)

        # take the first 10 s of the waveform
        len_waveform = int(len(waveform) / sr)
        len_waveform = min(len_waveform, 10)
        waveform = waveform[:sr * len_waveform]

        waveform = waveform[None, None, ...]  # [1, 1, 32000 * duration]

        with torch.no_grad():
            captions = model.generate_text(waveform=waveform)

        # split out music caption from the generated results
        description = captions[0]
        description = description.strip()
        print(f'Generated caption: [{description}]', file=sys.stderr)

        pred_caption[music_id] = description
        gt_caption[music_id] = music_description_list[count]

        if visualize_result:
            music_filename = "%s.mp3" % music_id
            music_path = os.path.join(music_save_path, music_filename)
            try:
                sf.write(music_path, waveform.squeeze(), 32000)
            except Exception as e:
                print(e)
                continue

        count += 1

    json.dump(pred_caption, f_gen, indent=4)
    json.dump(gt_caption, f_gt, indent=4)
    f_gt.close()
    f_gen.close()

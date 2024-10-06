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
from unimumo.util import load_model_from_config


# randomly pair each music with several aligned motion


def main(args):
    # data paths and save paths
    music_dir = 'data/music/audios'
    music_metadata_dir = 'data/music'
    motion_dir = 'data/motion'
    feature_dir = 'data/music/music_beat'
    motion_feature_save_dir = 'data/motion/aligned_motion_code'
    dataset_name = 'music4all'
    # model paths
    ckpt = '../My_Project/pretrained/motion_vqvae.ckpt'
    yaml_dir = 'configs/train_motion_vqvae.yaml'

    # set the fps of motion vqvae, should be 20 or 60
    fps = 60
    # set how many motion is paired for each music
    num_pair_each_music = 5
    # non-humanml3d data repeat time: determine the ratio of humanml3d and other data
    repeat_time = 30

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
    motion_data = {'train': [], 'val': [], 'test': []}
    aist = []
    dancedb = []
    humanml3d = []
    motion_mean = np.load(pjoin(motion_dir, 'Mean.npy'))
    motion_std = np.load(pjoin(motion_dir, 'Std.npy'))

    with cs.open(pjoin(motion_dir, 'ignore_list.txt'), "r") as f:
        for line in f.readlines():
            ignore.append(line.strip())
    for split in ['train', 'val', 'test']:
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
                for _ in range(repeat_time):
                    motion_data[split].append(line.strip())
                aist.append(line.strip())
        with cs.open(pjoin(motion_dir, f'dancedb_{split}.txt'), "r") as f:
            for line in f.readlines():
                if line.strip() in ignore:
                    continue
                if not os.path.exists(pjoin(motion_dir, split, 'joint_vecs', line.strip() + '.npy')):
                    continue
                for _ in range(repeat_time):
                    motion_data[split].append(line.strip())
                dancedb.append(line.strip())

    # load music data
    music_data = []
    for split in ['train', 'val', 'test']:
        with cs.open(pjoin(music_metadata_dir, f'{dataset_name}_{split}.txt'), "r") as f:
            for line in f.readlines():
                if not os.path.exists(pjoin(music_dir, line.strip() + '.wav')) and \
                   not os.path.exists(pjoin(music_dir, line.strip() + '.mp3')):
                    continue
                if os.path.exists(pjoin(feature_dir, line.strip() + '.pth')):
                    music_data.append([line.strip(), split])

    start_idx = int(args.start * len(music_data))
    end_idx = int(args.end * len(music_data))
    music_data = music_data[start_idx:end_idx]

    print(f'total music: {len(music_data)}')
    num_motion = len(motion_data['train']) + len(motion_data['val']) + len(motion_data['test'])
    print(f'total motion: {num_motion}')

    print(f'length of dance: {len(dancedb) + len(aist)}')
    print(f'length of non dance: {len(humanml3d)}')

    for data_idx, (music_id, split) in enumerate(music_data):
        # check whether the music has already been paired
        generated_motion_list = os.listdir(motion_feature_save_dir)
        generated_motion_list = [f for f in generated_motion_list if f[:len(music_id)] == music_id]
        if len(generated_motion_list) >= num_pair_each_music:
            print(f'{data_idx + 1}/{len(music_data)} already exists!')
            continue
        print(f'{data_idx + 1}/{len(music_data)}', end=' ')

        music_path = pjoin(music_dir, f'{music_id}.wav')
        if not os.path.exists(music_path):
            music_path = pjoin(music_dir, f'{music_id}.mp3')
        waveform, sr = librosa.load(music_path, sr=32000)
        music_duration = 30
        max_motion_length = int(music_duration * fps)
        max_music_length = int(music_duration * 32000)

        waveform = torch.FloatTensor(waveform)
        # cut or pad to max_music_length
        if waveform.shape[0] != max_music_length:
            if waveform.shape[0] > max_music_length:
                waveform = waveform[:max_music_length]
            else:
                zero_pad = torch.zeros(max_music_length)
                zero_pad[:waveform.shape[0]] = waveform
                waveform = zero_pad
        waveform = waveform[None, None, ...]
        print(f'music shape: {waveform.shape}', end=' ')

        # load music beat
        feature_dict = torch.load(pjoin(feature_dir, f'{music_id}.pth'))

        for pair_idx in range(num_pair_each_music):
            while True:
                mbeat = feature_dict['beat']

                random_motion_idx = random.randint(0, len(motion_data[split]) - 1)
                motion_name = motion_data[split][random_motion_idx]

                # load motion, and cut or repeat to match the target motion length
                motion = np.load(pjoin(motion_dir, split, 'joint_vecs', motion_name + '.npy'))
                if motion_name in humanml3d or motion_name in dancedb:
                    if fps == 60:  # interpolate the 20 fps data to 60 fps
                        motion = torch.Tensor(motion)
                        motion = rearrange(motion, 't d -> d t')
                        motion = torch.nn.functional.interpolate(motion[None, ...], scale_factor=3, mode='linear')
                        motion = rearrange(motion[0], 'd t -> t d').numpy()

                    motion_length = motion.shape[0]

                    aug = max_motion_length // motion_length
                    if aug < 1:  # if loaded motion is longer than target length, then randomly cut
                        start_idx = random.randint(0, motion_length - max_motion_length)
                        motion = motion[start_idx:start_idx + max_motion_length]
                    elif aug == 1:
                        if max_motion_length - motion_length <= 2.5 * fps:  # loaded motion is roughly
                            motion = motion                          # the same length as target length
                        else:
                            motion = np.tile(motion, (2, 1))  # repeat motion two times
                    else:  # if target length is more than 2 times longer than loaded motion, then repeat motion
                        max_repeat = aug
                        if max_motion_length - max_repeat * motion.shape[0] > 2.5 * fps:
                            max_repeat += 1
                        motion = np.tile(motion, (max_repeat, 1))
                else:  # if motion is from AIST++
                    if fps == 20:
                        motion = motion[::3]  # 60 fps -> 20 fps

                    motion_length = motion.shape[0]

                    if max_motion_length // motion_length < 1:  # if loaded motion is longer than target length,
                        start_idx = random.randint(0, motion_length - max_motion_length)  # then randomly cut
                        motion = motion[start_idx:start_idx + max_motion_length]
                    elif max_motion_length // motion_length == 1:  # loaded motion is roughly
                        pass                                       # the same length as target length
                    else:  # if target length is more than 2 times longer than loaded motion, then repeat motion
                        max_repeat = max_motion_length // motion_length + 1
                        motion = np.tile(motion, (max_repeat, 1))

                # scale mbeat to 20 or 60 fps
                scale_ratio = 32000 / fps
                mbeat = (mbeat / scale_ratio).numpy()
                mbeat = (np.rint(mbeat)).astype(int)

                try:
                    # get motion visual beats
                    rec_ric_data = motion_process.recover_from_ric(torch.from_numpy(motion).unsqueeze(0).float(), 22)
                    skel = rec_ric_data.squeeze().numpy()
                    directogram, vimpact = visual_beat.calc_directogram_and_kinematic_offset(skel)
                    peakinds, peakvals = visual_beat.get_candid_peaks(vimpact, sampling_rate=fps)
                    tempo_bpms, result = visual_beat.getVisualTempogram(vimpact, window_length=4, sampling_rate=fps)
                    visual_beats = visual_beat.find_optimal_paths(
                        list(map(lambda x, y: (x, y), peakinds, peakvals)), result, sampling_rate=fps
                    )
                    # turn visual beats into binary
                    vbeats = np.zeros((skel.shape[0]))
                    if len(visual_beats) != 0:
                        for beat in visual_beats[0]:
                            idx = beat[0]
                            vbeats[idx] = 1
                except Exception as e:
                    print(e)
                    continue

                # turn music beats also into binary
                mbeats = np.zeros(max_motion_length)
                for beat in mbeat:
                    if beat < len(mbeats):
                        mbeats[beat] = 1

                try:
                    alignment = dtw(vbeats, mbeats, keep_internals=True, step_pattern=rabinerJuangStepPattern(6, "d"))
                    wq = warp(alignment, index_reference=False)
                    final_motion = interpolation.interp(motion, wq)
                    break
                except Exception as e:  # if alignment fails, try a new one
                    print(e)
                    continue

            motion = (final_motion - motion_mean) / motion_std
            motion = torch.FloatTensor(motion)  # T, D

            # pad the motion into the same shape
            if motion.shape[0] != max_motion_length:
                if motion.shape[0] > max_motion_length:
                    motion = motion[:max_motion_length, :]
                else:
                    zero_pad = torch.zeros((max_motion_length, 263))
                    zero_pad[:motion.shape[0], :] = motion
                    motion = zero_pad

            motion = motion[None, ...]

            zero_waveform = torch.zeros_like(waveform)

            music_emb, motion_emb = model.encode(zero_waveform.to(device), motion.to(device))
            motion_code = model.quantizer.encode(motion_emb)
            motion_token = motion_code.squeeze()

            print(f'motion {pair_idx} shape {motion.shape}, feature shape {motion_token.shape}', end='; ')
            motion_token = motion_token.cpu()

            motion_token_save_path = pjoin(motion_feature_save_dir, music_id + f'_!motion_code!_{motion_name}.pth')
            torch.save(motion_token, motion_token_save_path)  # 4, 1500
        print('')


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

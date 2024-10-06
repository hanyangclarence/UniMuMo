import os
import torch
from omegaconf import OmegaConf
from madmom.features.downbeats import DBNDownBeatTrackingProcessor as DownBproc
import codecs as cs
from os.path import join as pjoin
import librosa
import argparse

import sys
from pathlib import Path
# Get the directory of the current script
current_dir = Path(__file__).parent
# Get the parent directory
parent_dir = current_dir.parent
# Add the parent directory to sys.path
sys.path.append(str(parent_dir))

from unimumo.audio.audiocraft_.models.builders import get_compression_model
from unimumo.audio.beat_detection.test_beat_detection import get_music_beat, build_beat_tracker


def main(args):
    data_dir = "data/music/audios"
    meta_dir = "data/music"
    code_dir_name = 'music_code'
    beat_dir_name = 'music_beat'
    save_dir = "data/music"
    vqvae_ckpt_path = 'pretrained/music_vqvae.bin'
    beat_tracker_ckpt_path = 'unimumo/audio/beat_detection/pretrained_models/baseline_v1'

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    beat_tracker_config = {
          'model_type': 'bsl_blstm',
          'model_dir': 'unimumo/audio/beat_detection/pretrained_models/baseline_v1',
          'model_simpname': 'baseline_v1',
          'num_tempi': 60,
          'transition_lambda': 140,
          'observation_lambda': 8,
          'threshold': 0.55,
          'f_measure_threshold': 0.07,
          'beats_per_bar': [ 3, 4 ],
          'max_bpm': 215,
          'min_bpm': 55,
          'fps': 100,
    }

    duration = 10
    sr = 32000

    # load vqvae model
    pkg = torch.load(vqvae_ckpt_path, map_location='cpu')
    cfg = OmegaConf.create(pkg['xp.cfg'])
    model = get_compression_model(cfg)
    model.load_state_dict(pkg['best_state'])
    model.eval()
    model.to(device)

    # load beat detection model
    beat_detection_model = build_beat_tracker(beat_tracker_ckpt_path)
    hmm_proc = DownBproc(**beat_tracker_config)
    music_target_length = int(duration * sr)

    # prepare for data
    music_data = []
    for split in ['train', 'test', 'val']:
        with cs.open(pjoin(meta_dir, f'music4all_{split}.txt'), 'r') as f:
            for line in f.readlines():
                music_data.append(line.strip())

    # prepare for save dir
    code_dir = pjoin(save_dir, code_dir_name)
    beat_dir = pjoin(save_dir, beat_dir_name)
    os.makedirs(code_dir, exist_ok=True)
    os.makedirs(beat_dir, exist_ok=True)

    # select a portion of the total data to process
    music_data.sort()
    start_idx = int(args.start * len(music_data))
    end_idx = int(args.end * len(music_data))
    music_data = music_data[start_idx:end_idx]

    with torch.no_grad():
        # traverse the data
        for i, music_id in enumerate(music_data):
            code_save_path = pjoin(code_dir, music_id + '.pth')
            beat_save_path = pjoin(beat_dir, music_id + '.pth')
            if os.path.exists(code_save_path) and os.path.exists(beat_save_path):
                print('%s.pth exists' % music_id)
                continue

            if os.path.exists(pjoin(data_dir, f'{music_id}.mp3')):
                music_path = pjoin(data_dir, f'{music_id}.mp3')
            elif os.path.exists(pjoin(data_dir, f'{music_id}.wav')):
                music_path = pjoin(data_dir, f'{music_id}.wav')
            else:
                print(f"music {music_id} not found")
                continue

            beat, bpm = get_music_beat(
                music_pth=music_path,
                rnn=beat_detection_model,
                hmm_proc=hmm_proc,
                device=device
            )
            if len(beat) == 0:
                print('music ' + str(music_id) + ' have failed beat detection with len 0')
                continue

            beat = (beat * sr).astype(int)

            if beat[-1] - beat[0] < music_target_length:
                print('music ' + str(music_id) + ' have failed beat detection')
                continue

            waveform, sr = librosa.load(music_path, sr=sr)

            # convert to tensor
            beat = torch.LongTensor(beat)
            waveform = torch.FloatTensor(waveform)

            # extract feature
            x = waveform[None, None, ...].to(device)

            codes, scale = model.encode(x)

            code_data = {'codes': codes.cpu()}
            torch.save(code_data, code_save_path)

            beat_data = {'beat': beat}
            torch.save(beat_data, beat_save_path)

            print('%d/%d, %s processed, bpm: %d' % (i+1, len(music_data), music_id, bpm))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-s',
        '--start',
        type=float,
        required=False,
        default=0.0,
        help='the start ratio for this preprocessing'
    )
    parser.add_argument(
        '-e',
        '--end',
        type=float,
        required=False,
        default=1.0,
        help='the end ratio of this preprocessing'
    )
    args = parser.parse_args()

    main(args)

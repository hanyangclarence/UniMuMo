import torch
import os
import numpy as np
import codecs as cs
from torch.utils.data import Dataset
from os.path import join as pjoin
import random
import pickle
import librosa
import typing as tp
from einops import rearrange

from unimumo.alignment import visual_beat, interpolation
from unimumo.motion import motion_process
from dtw import *


class MusicMotionDataset(Dataset):
    def __init__(
        self, split: str, music_dir: str, motion_dir: str, music_meta_dir: str, music_beat_dir: str,
        duration: int = 10, use_humanml3d: bool = False, music_dataset_name: str = 'music4all',
        traverse_motion: bool = True, align: bool = True, music_ignore_name: tp.Optional[str] = None,
        motion_fps: int = 20, dance_repeat_time: int = 1
    ):
        self.split = split

        # all data paths
        self.music_dir = music_dir
        self.motion_dir = motion_dir
        self.music_meta_dir = music_meta_dir
        self.music_beat_dir = music_beat_dir

        # about data loading settings
        self.traverse_motion = traverse_motion
        self.align = align
        self.duration = duration

        # about motion settings
        self.njoints = 22
        self.fps = motion_fps
        assert motion_fps in [20, 60], f"motion fps can only be 20 or 60, input {motion_fps}"
        self.motion_dim = 263
        self.max_motion_length = duration * self.fps

        # about music settings
        self.vqvae_sr = 32000
        self.music_target_length = int(duration * 50)

        # all data lists
        self.humanml3d = []
        self.aist = []
        self.dancedb = []
        self.motion_data = []
        self.motion_ignore = []

        self.music_data = []
        self.music_ignore = []

        # load motion data
        with cs.open(pjoin(self.motion_dir, 'ignore_list.txt'), "r") as f:
            for line in f.readlines():
                self.motion_ignore.append(line.strip())
        with cs.open(pjoin(self.motion_dir, f'humanml3d_{self.split}.txt'), "r") as f:
            for line in f.readlines():
                if not os.path.exists(pjoin(self.motion_dir, self.split, 'joint_vecs', line.strip() + '.npy')):
                    continue
                self.humanml3d.append(line.strip())
        with cs.open(pjoin(self.motion_dir, f'aist_{self.split}.txt'), "r") as f:
            for line in f.readlines():
                if line.strip() in self.motion_ignore:
                    continue
                if not os.path.exists(pjoin(self.motion_dir, self.split, 'joint_vecs', line.strip() + '.npy')):
                    continue
                self.motion_data.append(line.strip())
                self.aist.append(line.strip())
        with cs.open(pjoin(self.motion_dir, f'dancedb_{self.split}.txt'), "r") as f:
            for line in f.readlines():
                if not os.path.exists(pjoin(self.motion_dir, self.split, 'joint_vecs', line.strip() + '.npy')):
                    continue
                self.motion_data.append(line.strip())
                self.dancedb.append(line.strip())
        print(f'Humanml3d size: {len(self.humanml3d)}, dance size: {len(self.motion_data)}')

        if use_humanml3d:
            self.motion_data = self.motion_data * dance_repeat_time + self.humanml3d
        # load motion mean, std and length
        self.mean = np.load(pjoin(self.motion_dir, 'Mean.npy'))
        self.std = np.load(pjoin(self.motion_dir, 'Std.npy'))
        with open(pjoin(self.motion_dir, f'{self.split}_length.pickle'), 'rb') as f:
            self.length = pickle.load(f)
        print(f"Total number of motions: {len(self.motion_data)}")

        # load music data
        if music_ignore_name is None:
            music_ignore_name = f'{music_dataset_name}_ignore.txt'
        with cs.open(pjoin(self.music_meta_dir, music_ignore_name), "r") as f:
            for line in f.readlines():
                self.music_ignore.append(line.strip())
        with cs.open(pjoin(self.music_meta_dir, f'{music_dataset_name}_{self.split}.txt'), "r") as f:
            for line in f.readlines():
                if line.strip() in self.music_ignore:
                    continue
                if not os.path.exists(pjoin(self.music_dir, line.strip() + '.wav')):
                    continue
                if not os.path.exists(pjoin(self.music_beat_dir, line.strip() + '.pth')):
                    continue
                self.music_data.append(line.strip())
        print(f"Total number of music: {len(self.music_data)}")

    def __len__(self):
        if self.traverse_motion:
            return len(self.motion_data)
        else:
            return len(self.music_data)

    def __getitem__(self, idx):
        if self.traverse_motion:  # sequentially load motion, randomly pair with music
            motion_name = self.motion_data[idx]
            random_music_idx = random.randint(0, len(self.music_data) - 1)
            music_name = self.music_data[random_music_idx]
        else:  # sequentially load music, randomly pair with motion
            music_name = self.music_data[idx]
            random_motion_idx = random.randint(0, len(self.motion_data) - 1)
            motion_name = self.motion_data[random_motion_idx]

        # load motion, and cut or repeat to match the target motion length
        motion = np.load(pjoin(self.motion_dir, self.split, 'joint_vecs', motion_name + '.npy'))
        if motion_name in self.humanml3d or motion_name in self.dancedb:
            if self.fps == 60:  # interpolate the 20 fps data to 60 fps
                motion = torch.Tensor(motion)
                motion = rearrange(motion, 't d -> d t')
                motion = torch.nn.functional.interpolate(motion[None, ...], scale_factor=3, mode='linear')
                motion = rearrange(motion[0], 'd t -> t d').numpy()
                motion_length = self.length[motion_name] * 3
            else:
                motion_length = self.length[motion_name]

            aug = self.max_motion_length // motion_length
            if aug < 1:  # if loaded motion is longer than target length, then randomly cut
                start_idx = random.randint(0, motion_length - self.max_motion_length)
                motion = motion[start_idx:start_idx+self.max_motion_length]
            elif aug == 1:
                if self.max_motion_length - motion_length <= 2.5 * self.fps:  # loaded motion is roughly
                    motion = motion                               # the same length as target length
                else:
                    motion = np.tile(motion, (2, 1))  # repeat motion two times
            else:  # if target length is more than 2 times longer than loaded motion, then repeat motion
                max_repeat = aug
                if self.max_motion_length - max_repeat*motion.shape[0] > 2.5 * self.fps:
                    max_repeat += 1
                motion = np.tile(motion, (max_repeat, 1))
        else:  # if motion is from AIST++
            if self.fps == 20:  # down sample 60 fps to 20 fps
                motion = motion[::3]
                motion_length = self.length[motion_name] // 3
            else:
                motion_length = self.length[motion_name]

            if self.max_motion_length // motion_length < 1:  # if loaded motion is longer than target length,
                start_idx = random.randint(0, motion_length - self.max_motion_length)  # then randomly cut
                motion = motion[start_idx:start_idx + self.max_motion_length]
            elif self.max_motion_length // motion_length == 1:  # loaded motion is roughly
                pass                                            # the same length as target length
            else:  # if target length is more than 2 times longer than loaded motion, then repeat motion
                max_repeat = self.max_motion_length // motion_length + 1
                motion = np.tile(motion, (max_repeat, 1))

        # load music
        music_path = pjoin(self.music_dir, f'{music_name}.wav')
        waveform, sr = librosa.load(music_path, sr=self.vqvae_sr)  # [32000 x T]
        waveform_target_length = int(self.duration * self.vqvae_sr)
        # load music beat
        beat_dict = torch.load(pjoin(self.music_beat_dir, f'{music_name}.pth'))
        mbeat = beat_dict['beat']

        # random cut waveform and music beat
        start_idx = random.randint(0, waveform.shape[-1] - waveform_target_length)
        end_idx = start_idx + waveform_target_length
        waveform = waveform[start_idx:end_idx]
        mbeat = mbeat[torch.where((start_idx <= mbeat) & (mbeat <= end_idx))[0]]
        mbeat = mbeat - start_idx

        if self.align:  # align music and motion using dynamic time wrapping
            # scale mbeat to 20 or 60 fps
            scale_ratio = self.vqvae_sr / self.fps
            mbeat = (mbeat / scale_ratio).numpy()
            mbeat = (np.rint(mbeat)).astype(int)

            # get motion visual beats
            rec_ric_data = motion_process.recover_from_ric(torch.from_numpy(motion).unsqueeze(0).float(), self.njoints)
            skel = rec_ric_data.squeeze().numpy()
            directogram, vimpact = visual_beat.calc_directogram_and_kinematic_offset(skel)
            peakinds, peakvals = visual_beat.get_candid_peaks(vimpact, sampling_rate=self.fps)
            tempo_bpms, result = visual_beat.getVisualTempogram(vimpact, window_length=4, sampling_rate=self.fps)
            visual_beats = visual_beat.find_optimal_paths(
                list(map(lambda x, y: (x, y), peakinds, peakvals)), result, sampling_rate=self.fps
            )
            # turn visual beats and music beats into binary
            vbeats = np.zeros((skel.shape[0]))
            if len(visual_beats) != 0:
                for beat in visual_beats[0]:
                    idx = beat[0]
                    vbeats[idx] = 1
            mbeats = np.zeros((self.duration * self.fps))
            for beat in mbeat:
                if beat < len(mbeats):
                    mbeats[beat] = 1

            try:
                alignment = dtw(vbeats, mbeats, keep_internals=True, step_pattern=rabinerJuangStepPattern(6, "d"))
                wq = warp(alignment, index_reference=False)
                final_motion = interpolation.interp(motion, wq)
            except Exception as e:  # if alignment fails, then just use the original motion
                print(e)
                final_motion = motion
        else:
            final_motion = motion

        motion = (final_motion - self.mean) / self.std

        waveform = torch.FloatTensor(waveform)[None, ...]
        motion = torch.FloatTensor(motion)
        waveform = waveform.to(memory_format=torch.contiguous_format)
        motion = motion.to(memory_format=torch.contiguous_format)

        # if alignment fails, the motion length may not match the target motion length
        # then we need to cut or pad
        if motion.shape[0] != self.max_motion_length:
            if motion.shape[0] > self.max_motion_length:
                motion = motion[:self.max_motion_length, :]
            else:
                zero_pad = torch.zeros((self.max_motion_length, self.motion_dim))
                zero_pad[:motion.shape[0], :] = motion
                motion = zero_pad

        return {
            'motion': motion,  # [fps x T, 263]
            'waveform': waveform,  # [1, 32000 x T]
        }


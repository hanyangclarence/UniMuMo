import torch
import os
import numpy as np
import codecs as cs
from torch.utils.data import Dataset
from os.path import join as pjoin
import random
import pandas as pd
import json


# Experiment
# Also load the gpt generated text

class MusicMotionCaptioningDataset(Dataset):
    def __init__(
        self, split, music_meta_dir, motion_meta_dir,
        music_code_dir, motion_code_dir,
        duration=10,
        vqvae_sr=32000,
        dropout_prob=0.1,
        music_dataset_name='music4all',
        ignore_file_name='music4all_ignore.txt',
        natural_language_caption_ratio=0.3,
        use_mullama=True,
        humanml3d_only=False,
    ):
        # all data paths
        self.motion_meta_dir = motion_meta_dir
        self.music_meta_dir = music_meta_dir
        self.music_code_dir = music_code_dir
        self.motion_code_dir = motion_code_dir

        # settings about data loading
        self.split = split
        self.duration = duration
        self.vqvae_sr = vqvae_sr
        self.music_target_length = int(duration * 50)
        self.dropout_prob = dropout_prob
        self.natural_language_caption_ratio = natural_language_caption_ratio

        # all data lists
        self.music_data = []
        self.motion_data = []
        self.music_ignore_list = []

        self.humanml3d = []
        self.aist = []
        self.dancedb = []

        # load data related to text descriptions
        # load metadata of music4all
        self.text_df = pd.read_csv(pjoin(self.music_meta_dir, 'music4all_metadata.csv'), index_col=0)
        # load mu-llama generated text descriptions
        with open(pjoin(self.music_meta_dir, 'music4all_captions_mullama.json'), 'r') as caption_fd:
            llama_music_caption = json.load(caption_fd)
        # load gpt generated text descriptions
        with open(pjoin(self.music_meta_dir, 'music4all_captions_gpt.json'), 'r') as caption_fd:
            self.music_caption = json.load(caption_fd)
        # merge two descriptions
        if use_mullama:
            for k, v in llama_music_caption.items():
                if k in self.music_caption.keys():
                    if 'male vocalist' in v:
                        continue
                    self.music_caption[k].append(v)
        # load humanml3d text descriptions
        humanml3d_text_dir = pjoin(self.motion_meta_dir, 'humanml3d_text_description')
        humanml3d_descriptions = os.listdir(humanml3d_text_dir)
        self.humanml3d_text = {}
        for desc_txt in humanml3d_descriptions:
            with open(pjoin(self.motion_meta_dir, 'humanml3d_text_description', desc_txt), 'r', encoding='UTF-8') as f:
                texts = []
                lines = f.readlines()
                for line in lines:
                    text = line.split('#')[0]
                    if text[-1] == '.':
                        text = text[:-1]
                    texts.append(text)
                self.humanml3d_text[desc_txt.split('.')[0]] = texts
        # genre mapping for aist
        self.aist_genre_map = {
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

        # load motion mean and std for normalization
        self.mean = np.load(pjoin(self.motion_meta_dir, 'Mean.npy'))
        self.std = np.load(pjoin(self.motion_meta_dir, 'Std.npy'))

        # load all paired motion codes
        motion_data_full = os.listdir(self.motion_code_dir)
        motion_data_full = ['.'.join(s.split('.')[:-1]) for s in motion_data_full]  # remove the .pth at the end

        # load motion filenames
        with cs.open(pjoin(self.motion_meta_dir, f'humanml3d_{self.split}.txt'), "r") as f:
            for line in f.readlines():
                self.humanml3d.append(line.strip())
                if line.strip() in motion_data_full:
                    self.motion_data.append(line.strip())
        with cs.open(pjoin(self.motion_meta_dir, f'aist_{self.split}.txt'), "r") as f:
            for line in f.readlines():
                self.aist.append(line.strip())
                if line.strip() in motion_data_full:
                    self.motion_data.append(line.strip())
        with cs.open(pjoin(self.motion_meta_dir, f'dancedb_{self.split}.txt'), "r") as f:
            for line in f.readlines():
                self.dancedb.append(line.strip())
                if line.strip() in motion_data_full:
                    self.motion_data.append(line.strip())
        print(f'Humanml3d size: {len(self.humanml3d)}, aist size: {len(self.aist)}, dancedb size: {len(self.dancedb)}')
        print(f'Total motion data in {split}: {len(self.motion_data)}')

        # use only humanml3d data if speficied
        if humanml3d_only:
            self.motion_data = [s for s in self.motion_data if s in self.humanml3d]
            music_with_paired_motion = list(set([s.split('_!motion_code!_')[0] for s in self.motion_data]))
            print(f"Humanml3d only: Total number of motion {len(self.motion_data)}")
            print(f'Humanml3d only: Total number of music with paired motion data {len(music_with_paired_motion)}')

        # load music filenames
        with cs.open(pjoin(self.music_meta_dir, ignore_file_name), "r") as f:
            for line in f.readlines():
                self.music_ignore_list.append(line.strip())
        with cs.open(pjoin(self.music_meta_dir, f'{music_dataset_name}_{self.split}.txt'), "r") as f:
            for line in f.readlines():
                if line.strip() in self.music_ignore_list:
                    continue
                if not os.path.exists(pjoin(self.music_code_dir, line.strip() + '.pth')):
                    continue
                if line.strip() not in self.music_caption.keys():
                    continue
                self.music_data.append(line.strip())
        # make sure that all music data have captions
        self.music_data = [s for s in self.music_data if s in self.music_caption.keys()]

        print(f'Total number of music in {split} set: {len(self.music_data)}')

    def __len__(self):
        return len(self.music_data)

    def __getitem__(self, idx):
        music_id = self.music_data[idx]

        # load music token
        music_code = torch.load(pjoin(self.music_code_dir, f'{music_id}.pth'))['codes'][0]  # 4, T

        # load motion token
        motion_name = random.choice(self.motion_data)  # randomly choose a paired motion
        motion_code = torch.load(pjoin(self.motion_code_dir, motion_name + '.pth'))  # 4, T

        # random cut waveform and music beat
        start_idx = random.randint(0, music_code.shape[-1] - self.music_target_length - 2)
        end_idx = start_idx + self.music_target_length
        music_code = music_code[:, start_idx:end_idx]

        # music text prompt construction
        # use llama caption
        music_text_prompt = random.choice(self.music_caption[music_id])

        # construct motion text prompt
        motion_description = ''
        if motion_name in self.dancedb:
            feeling = motion_name.split('_')[1]  # the feeling of the dance
            desc_choices = [f'This is a {feeling} dance.', f'The dance is {feeling}.']
            motion_description = random.choice(desc_choices)
        elif motion_name in self.aist:
            genre_id = motion_name.split('_')[0]
            genre = self.aist_genre_map[genre_id]
            desc_choices = [f'The genre of the dance is {genre}.', f'The style of the dance is {genre}.',
                            f'This is a {genre} style dance.']
            motion_description = random.choice(desc_choices)
        elif motion_name in self.humanml3d:
            text_choice = self.humanml3d_text[motion_name]
            desc = random.choice(text_choice)
            desc_choices = [f'The motion is that {desc}.', f'The dance is that {desc}.']
            motion_description = random.choice(desc_choices)
        else:
            ValueError()

        # text_prompt = music_text_prompt.capitalize() + ' ' + motion_description.capitalize()
        # Here I change to use tokens to separate them
        text_prompt = music_text_prompt.capitalize() + ' <separation> ' + motion_description.capitalize()

        if idx % 50 == 0:
            print(f'{music_code.shape}, {motion_code.shape}, {text_prompt}')

        return {
            'motion_code': motion_code,  # (4, 500)
            'music_code': music_code,  # (4, 500)
            'text': text_prompt
        }

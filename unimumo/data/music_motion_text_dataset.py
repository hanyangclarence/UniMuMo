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

class MusicMotionTextDataset(Dataset):
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
        self.motion_data = os.listdir(self.motion_code_dir)
        self.motion_data = ['.'.join(s.split('.')[:-1]) for s in self.motion_data]  # remove the .pth at the end
        music_with_paired_motion = list(set([s.split('_!motion_code!_')[0] for s in self.motion_data]))  # find all music that are paired with motion
        print(f"Total number of motion {len(self.motion_data)}")
        print(f'Total number of music with paired motion data {len(music_with_paired_motion)}')

        # load motion filenames
        with cs.open(pjoin(self.motion_meta_dir, f'humanml3d_{self.split}.txt'), "r") as f:
            for line in f.readlines():
                self.humanml3d.append(line.strip())
        with cs.open(pjoin(self.motion_meta_dir, f'aist_{self.split}.txt'), "r") as f:
            for line in f.readlines():
                self.aist.append(line.strip())
        with cs.open(pjoin(self.motion_meta_dir, f'dancedb_{self.split}.txt'), "r") as f:
            for line in f.readlines():
                self.dancedb.append(line.strip())
        print(f'Humanml3d size: {len(self.humanml3d)}, aist size: {len(self.aist)}, dancedb size: {len(self.dancedb)}')

        # use only humanml3d data if speficied
        if humanml3d_only:
            self.motion_data = [s for s in self.motion_data if s.split('_!motion_code!_')[-1] in self.humanml3d]
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
                if line.strip() not in music_with_paired_motion:
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
        selection = [s for s in self.motion_data if s[:len(music_id)] == music_id]  # motion name that starts with music_id
        motion_name = random.choice(selection)  # randomly choose a paired motion
        motion_name = motion_name.split('_!motion_code!_')[1]
        motion_code = torch.load(pjoin(self.motion_code_dir, f'{music_id}_!motion_code!_{motion_name}.pth'))  # 4, T

        # random cut waveform and music beat
        start_idx = random.randint(0, music_code.shape[-1] - self.music_target_length - 2)
        end_idx = start_idx + self.music_target_length
        music_code = music_code[:, start_idx:end_idx]
        motion_code = motion_code[:, start_idx:end_idx]

        # music text prompt construction
        if random.uniform(0, 1) < self.natural_language_caption_ratio:
            # use llama caption
            music_text_prompt = random.choice(self.music_caption[music_id])
        else:
            # use constructed text prompt
            tag_list = self.text_df.loc[music_id, 'tags']
            if pd.isna(tag_list):
                tag_list = ['nan value']
                tag_is_empty = True
            else:
                tag_list = tag_list.split('\t')
                tag_is_empty = False

            # choose for tempo descriptor
            tempo = self.text_df.loc[music_id, 'tempo']
            if tempo < 60:
                s1 = ['extremely', 'very']
                s2 = ['slow', 'languid', 'lethargic', 'relaxed', 'leisure', 'chilled']
                tempo_description = f'{random.choice(s1)} {random.choice(s2)}'
            elif 60 <= tempo < 75:
                tempo_description = random.choice(['slow', 'languid', 'lethargic', 'relaxed', 'leisure', 'chilled'])
            elif 75 <= tempo < 110:
                tempo_description = random.choice(['moderate', 'easy-going', 'laid-back', 'medium', 'balanced', 'neutral'])
            elif 110 <= tempo < 150:
                tempo_description = random.choice(['fast', 'upbeat', 'high', 'brisk', 'quick', 'rapid', 'swift'])
            else:
                s1 = ['extremely', 'very', 'highly']
                s2 = ['fast', 'upbeat', 'high', 'brisk', 'quick', 'rapid', 'swift']
                tempo_description = f'{random.choice(s1)} {random.choice(s2)}'

            # choose for energy descriptor
            energy = self.text_df.loc[music_id, 'energy']
            if energy < 0.1:
                s1 = ['extremely', 'very']
                s2 = ['soft', 'calm', 'peaceful', 'serene', 'gentle', 'light', 'tranquil', 'mild', 'mellow']
                energy_description = f'{random.choice(s1)} {random.choice(s2)}'
            elif 0.1 <= energy < 0.4:
                energy_description = random.choice(
                    ['soft', 'calm', 'peaceful', 'serene', 'gentle', 'light', 'tranquil', 'mild', 'mellow'])
            elif 0.4 <= energy < 0.7:
                energy_description = random.choice(['moderate', 'comfortable', 'balanced', 'relaxing'])
            elif 0.7 <= energy < 0.95:
                energy_description = random.choice(
                    ['intense', 'powerful', 'strong', 'vigorous', 'fierce', 'potent', 'energetic'])
            else:
                s1 = ['extremely', 'very', 'highly']
                s2 = ['intense', 'powerful', 'strong', 'vigorous', 'fierce', 'potent', 'energetic']
                energy_description = f'{random.choice(s1)} {random.choice(s2)}'

            # drop out some tags
            p = random.uniform(0, 1)
            if p < self.dropout_prob:
                tempo_description = None
            p = random.uniform(0, 1)
            if p < self.dropout_prob:
                energy_description = None
            filtered_tag_list = []
            for tag in tag_list:
                p = random.uniform(0, 1)
                if p > self.dropout_prob:
                    filtered_tag_list.append(tag)

            # construct phrases
            noun_choices = ['tempo', 'speed', 'pace', 'BPM', 'rhythm', 'beat']
            tempo_choices = [f'with a {tempo_description} {random.choice(noun_choices)}',
                             f'whose {random.choice(noun_choices)} is {tempo_description}',
                             f'a {tempo_description} music', f'set in a {tempo_description} {random.choice(noun_choices)}']

            noun_choices = ['intensity', 'energy']
            energy_choices = [f'which is {energy_description}', f'with {energy_description} {random.choice(noun_choices)}',
                              f'a {energy_description} music',
                              f'whose {random.choice(noun_choices)} is {energy_description}']

            noun_choices = ['genre', 'style', 'type', 'category']
            if len(filtered_tag_list) == 0:
                filtered_tag_list = [random.choice(tag_list)]  # ensure at least have 1 tag
            if len(filtered_tag_list) == 1:
                tag_string = filtered_tag_list[0]
            else:
                tag_string = ', '.join(filtered_tag_list[:-1]) + f' and {filtered_tag_list[-1]}'
            tag_choices = [f'this is a track which is {tag_string}',
                           f'this song has the {random.choice(noun_choices)} of {tag_string}',
                           f'the music is {tag_string}', f'the {random.choice(noun_choices)} of the music is {tag_string}']

            phrase_tempo = random.choice(tempo_choices)
            phrase_energy = random.choice(energy_choices)
            phrase_tag = random.choice(tag_choices)

            text_prompt = []
            if tempo_description is not None:
                text_prompt.append(phrase_tempo)
            if energy_description is not None:
                text_prompt.append(phrase_energy)
            if not tag_is_empty:
                text_prompt.append(phrase_tag)

            random.shuffle(text_prompt)
            music_text_prompt = ', '.join(text_prompt) + '.'

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

        return {
            'motion_code': motion_code,  # (4, 500)
            'music_code': music_code,  # (4, 500)
            'text': text_prompt
        }

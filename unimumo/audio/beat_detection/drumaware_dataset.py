"""
this script was modified from the belowing source code:
    ## original source code: https://github.com/maxencemayrand/beat-tracking/blob/master/beatfinder/data.py


"""

import os
import numpy as np
import pandas as pd
import librosa
from torch.utils.data import Dataset
from pathlib import Path


import unimumo.audio.beat_detection.da_utils as utils


global_sr = 44100 

def beat2spec(beats, spec_timesteps = 1000, sr = 44100, hop_length = 441):
    """ purpose:
        1. convert beat information to shape of spec 
        2. convert beat, downbeat, nonbeat to label 2, 1, 0 """
        
    beat_label = np.zeros((spec_timesteps, 1))
    for (beat_time, beat_type) in beats:
        time_ind = int(beat_time*sr/hop_length)
        if str(int(beat_type)) == '1':
            beat_label[time_ind, 0] = 1
        else:
            beat_label[time_ind, 0] = 2
    return beat_label


def get_drum_feature(mix_feature_path, drumtype='OnlyDrum', aug_folder = 'datasets/sourcesep_aug'):
    """ given the path of mixture feature, 
        the corresponding drum/nondrum feature can be derived based on the folder organization """
        
    da_feature_path = mix_feature_path.replace('datasets/original', aug_folder)
    da_feature_path = os.path.join(Path(da_feature_path).parents[2], drumtype, *mix_feature_path.split('/')[-3:])
    if not os.path.exists(da_feature_path):
        print("can't find drum-aware feature :", da_feature_path)
        print("--> mix feature path:", mix_feature_path)
 
    return da_feature_path
    
class AudioBeats(object):
    r"""Basic class to represent a sample point.

    An `AudioBeats` object doesn't contain any data, but points to the right
    files and has many useful methods to precompute some of the data. They are
    meant to be the items of the pytorch Dataset class `AudioBeatsDataset` to
    train the `BeatFinder` model.

    An `AudioBeats` object represents a section of an audio file (possibly
    stretched) together with the beats (a sequence of times), the onsets, the
    subset of those onsets which are beats, and the spectrogram of the audio.
    The method `precompute` computes the spetrograms, onsets, and the onsets
    that are beats and store this information in files so that it can be quickly
    accessed during training.

    Arguments:
        audio_file (str): The relative path of the full audio file.
        beats_file (str): The relative path of the files containing the beats of
            `audio_file`. This is a `txt` file containing a single column of
            floating point numbers which are the times of the beat track (ground
            truth).
        feature_file (str): The relative path of the file where the input feature
            is stored upon calling `precompute_feature()`. This should be a `.npy`
            file. The enclosing directory will be created if it doesn't already
            exists.

        offset (float): The starting point of the sample in the stretched audio
            file.
        duration (float): The duration of the audio in seconds.
        length (int): The duration of the audio in samples (at the sampling rate
            determined in `constants.py`).
        song_duration (float): The duration (in seconds) of the full audio file.

        name (str): The name of the AudioBeats object.
    """

    def __init__(self,
                 audio_file,
                 beats_file,
                 feature_file,
                 offset,
                 duration,
                 length,
                 song_duration,
                 name):
        self.audio_file    = audio_file
        self.beats_file    = beats_file
        self.feature_file     = feature_file 
        #### get corresponding drum/non-drum feature paths
        self.nodrum_feature_file = get_drum_feature(self.feature_file, drumtype='NoDrum')
        self.drum_feature_file = get_drum_feature(self.feature_file, drumtype='OnlyDrum')
        
        self.offset        = offset        # starting point (in seconds) on the stretched wav file.
        self.duration      = duration      # duration in seconds
        self.length        = length        # same as duration, but is samples
        self.song_duration = song_duration # total duration of the stretched wav
        self.name          = name

    def get_wav(self):
        r"""Returns a numpy array of the audio section at the sampling rate
        determined by the `constants` module."""

        wav = librosa.load(self.audio_file,
                sr= global_sr, 
                offset=self.offset,
                duration=self.duration)[0]

        # This is to make sure that all samples have the same size so that we
        # can do minibatches.
        if len(wav) != self.length:
            z = np.zeros(self.length)
            m = min(len(wav), self.length)
            z[:m] = wav[:m]
            wav = z

        return wav

    def get_beats(self):
        r"""Returns a numpy array of the beats in seconds.
        """

        all_beats = np.loadtxt(self.beats_file)
        all_beats[:, 0] = all_beats[:, 0]
        mask = ( self.offset <= all_beats[:, 0]) & (all_beats[:, 0]<  self.offset +  self.duration)
        beats = all_beats[mask, :] 
        beats[:,0] = beats[:,0] - self.offset

        return beats

    def precompute_feature(self):
        r"""Compute the input feature and save in self.feature_file
        """

        if not os.path.exists(self.feature_file):
            path = os.path.dirname(self.feature_file)
            if not os.path.exists(path):
                os.makedirs(path)
            features = utils.madmom_feature(self.get_wav())
            np.save(self.feature_file, features)
        else:
            print("exists feature_file:", self.feature_file)

    def get_feature(self):
        r"""Returns the mel-scaled spectrogram (it must have been precomputed
        beforehand by calling `precompute_spec()` or `precompute()`)."""
    
        if not os.path.exists(self.feature_file):
            self.precompute_feature()
            
        return np.load(self.feature_file)
    
    def get_nodrum_feature(self):

        return np.load(self.nodrum_feature_file)

    
    def get_drum_feature(self):

        return np.load(self.drum_feature_file)


    def get_data(self):
        r"""
        Returns:
            features: input feature calculated using MadmomAPI, shape (1000 (frames), 314)
            beat_labels: beat/downbeat annotations in array with same shape as features
            nodrum_features: non-drum feature calculated using MadmomAPI, shape (1000 (frames), 314)
            drum_features: drum feature calculated using MadmomAPI, shape (1000 (frames), 314)
        """
        
        features = self.get_feature()
        nodrum_features = self.get_nodrum_feature()
        drum_features = self.get_drum_feature()
        beats = self.get_beats()
        beat_labels = beat2spec(beats)
        
        return features, beat_labels, nodrum_features, drum_features

   
class AudioBeatsDataset(Dataset):
    r"""
    Arguments:
        audiobeats_list (list): A list of AudioBeats objects. An
            `AudioBeatsDataset` object can be instantiated with either such a
            list or with a presaved `file` (see below).
    """

    def __init__(self, audiobeats_list):
        self.audiobeats_list = audiobeats_list

    def __len__(self):
        return len(self.audiobeats_list)

    def __getitem__(self, i):
        audiobeats = self.audiobeats_list[i]

        return audiobeats.get_data() # audiobeats  cyc 0707 modified

    def __add__(self, other):
        return ConcatAudioBeatsDataset([self, other])

    def precompute(self, mode='all', full=False):
        r"""Precomputes all the `AudioBeats` objects. This can take a
        substantial amount of time.

        """
        for j, audiobeats in enumerate(self.audiobeats_list):

            audiobeats.precompute_feature()

            print("precomputing input features ...")

    def save(self, file):
        r"""Save the dataset in a file. This is saved as a csv-style file where
        each row stores the information of an `AudioBeats` object (recall that
        those do not contain any actual data, but only link to portions of some
        files.)

        Arguments:
            file (str): The relative path of the file where to save the dataset.
                If the enclosing directory doesn't exist, it will be created.
        """

        path = os.path.dirname(file)
        if not os.path.exists(path):
            os.makedirs(path)

        df = pd.DataFrame()
        df['audio_file'] = [self.audiobeats_list[i].audio_file for i in range(len(self))]
        df['beats_file'] = [self.audiobeats_list[i].beats_file for i in range(len(self))]
        df['feature_file'] = [self.audiobeats_list[i].feature_file for i in range(len(self))]
        df['offset'] = [self.audiobeats_list[i].offset for i in range(len(self))]
        df['duration'] = [self.audiobeats_list[i].duration for i in range(len(self))]
        df['length'] = [self.audiobeats_list[i].length for i in range(len(self))]
        df['song_duration'] = [self.audiobeats_list[i].song_duration for i in range(len(self))]
        df['name'] = [self.audiobeats_list[i].name for i in range(len(self))]
        df.to_csv(file)
      

def load_dataset(file):
    r"""Return the `AudioBeatsDataset` saved in `file`.
    
    Argument:
        file (str): The relative path of a saved `AudioBeatsDataset` object,
            saved via the method `self.save`. An `AudioBeatsDataset` object can
            be instantiated with either such a file or with a list of
            `AudioBeats` objects (see above).
    Returns:
        dataset (AudioBeatsDataset): The dataset saved in `file`.
    """
    
    df = pd.read_csv(file, index_col=0)
    audiobeats_list = []
    for i in range(len(df)):
        audio_file    = df['audio_file'][i]
        beats_file    = df['beats_file'][i]
        feature_file     = df['feature_file'][i]
        offset        = df['offset'][i]
        duration      = df['duration'][i]
        length        = df['length'][i]
        song_duration = df['song_duration'][i]
        name          = df['name'][i]
        audiobeats = AudioBeats(audio_file, beats_file, feature_file,
                                 offset, duration, length, song_duration, name)
        audiobeats_list.append(audiobeats)
        
    dataset = AudioBeatsDataset(audiobeats_list)
    
    return dataset
        
class SubAudioBeatsDataset(AudioBeatsDataset):
    r"""Subset of an `AudioBeatsDataset` at specified indices.

    Arguments:
        dataset (AudioBeatsDataset): The original `AudioBeatsDataset`.
        indices (list): Selected indices in the original `AudioBeatsDataset`.
    """

    def __init__(self, dataset, indices):
        audiobeats_list = [dataset.audiobeats_list[i] for i in indices]
        super().__init__(audiobeats_list)


class AudioBeatsDatasetFromSong(AudioBeatsDataset):
    r"""An `AudioBeatsDataset` consisting of equally spaced `AudioBeats` of the
    same duration and completely covering a given audio file.

    Arguments:
        audio_file (str): The path of the audio file.
        beats_file (str): The path of the file containing all the beats (in
        seconds) in the audio file (a `.txt` file with one column of floats).
        precomputation_path (str): The directory where the data pointed by the
            `AudioBeats` objects is stored.
        duration (float): The duration of the audio pointed by each
            `AudioBeats`.
        stretch (float): The amount by which to stretch the audio.
        force_nb_samples (int or None): By default (if `None`) there will be as
            many `AudioBeats` as possible, side-by-side starting from time
            zero, until no `AudioBeats` can fit in the full audio. Hence, there
            might be a small portion (of length < duration) at the end not
            covered by any `AudioBeats`. If `force_nb_samples` is set to a
            larger number, there will be more `AudioBeats` with zero padding.
            This is useful when we have a large list of, e.g., ~30 seconds
            audio files and want to split each of them in three `AudioBeats`.
            We set `force_nb_samples = 3` so that even if an audio file is
            slightly less or slightly more than 30 seconds we always get 3
            samples (possibly with some zero padding on the right of the last
            one).
        song_offset (float): Starting point of the unstretched song.
        song_duration (float): Duration of the unstretched song.
    """

    def __init__(self, audio_file, beats_file, precomputation_path,
                  duration=10, force_nb_samples=None,
                 song_offset=None, song_duration=None):

        self.audio_file = audio_file
        self.song_name = os.path.splitext(os.path.basename(self.audio_file))[0]
        if song_duration:
            self.song_duration = song_duration 
        else:
            self.song_duration = librosa.get_duration(filename=self.audio_file) 
        if song_offset:
            self.song_offset = song_offset
        else:
            self.song_offset = 0
        self.precomputation_path = precomputation_path

        length = librosa.time_to_samples(duration, global_sr) 

        if force_nb_samples:
            nb_samples = force_nb_samples
        else:
            nb_samples = int(self.song_duration / duration)

        audiobeats_list = []
        for i in range(nb_samples):

            name = '{}.{:03d}'.format(self.song_name, i)

            feature_file = os.path.join(self.precomputation_path, 'cutfeatures/{}.npy'.format(name))
            offset = self.song_offset + i * duration
            audiobeats = AudioBeats(audio_file,
                                    beats_file,
                                    feature_file,
                                    offset,
                                    duration,
                                    length,
                                    self.song_duration,
                                    name)
            audiobeats_list.append(audiobeats)

        super().__init__(audiobeats_list)


class ConcatAudioBeatsDataset(AudioBeatsDataset):
    r"""Concatenate multiple `AudioBeatsDataset`s.

    Arguments:
        datasets (list): A list of `AudioBeatsDataset` objects.
    """

    def __init__(self, datasets):
           

        audiobeats_list = []
        for dataset in datasets:
            audiobeats_list += dataset.audiobeats_list

        super().__init__(audiobeats_list)


class AudioBeatsDatasetFromList(ConcatAudioBeatsDataset):
    r"""An `AudioBeatsDataset` instantiated from a file containing a list of
    audio files.

    Arguments:
        audio_files (str): The path of a `.txt` file where each line is the
            relative path of an audio file (relative to where `audio_files` is).
        precomputation_path (str): Where to store the precomputated data of each
            item.
        duration (float): Duration of each audio sample.
        force_nb_samples (int or None): To pass to `AudioBeatsDatasetFromSong`.
        audio_list: True if the input audio_files is an audio list instead of a `.txt` file.
        dataset_path: only required when audio_files is not a `.txt`
    """

    def __init__(self, audio_files, precomputation_path, duration=10, 
                  force_nb_samples=None, audio_list = False, dataset_path = None):
        ### using audio_files.txt for initialization
        if not audio_list:
            dataset_path = os.path.dirname(audio_files)
            beats_dir = os.path.join(dataset_path, 'downbeats/')
    
            datasets = []
            
            with open(audio_files) as f:
                for line in f.readlines():
                    relative_audio_file = os.path.normpath(line.strip('\n'))
                    audio_name = os.path.splitext(os.path.basename(relative_audio_file))[0]
                    audio_file = os.path.join(dataset_path, relative_audio_file)
                    beats_file = os.path.join(beats_dir, audio_name+'.beats')
                    dataset = AudioBeatsDatasetFromSong(audio_file, beats_file, precomputation_path,
                                                         duration, force_nb_samples)
                    datasets.append(dataset)
    
            super().__init__(datasets)
        ### using audio_files in list format for initialization
        else:
            dataset_path = dataset_path

            beats_dir = os.path.join(dataset_path, 'downbeats/')
    
            datasets = []
            for line in audio_files:
                relative_audio_file = os.path.normpath(line.strip('\n'))
                audio_name = os.path.splitext(os.path.basename(relative_audio_file))[0]
                audio_file = os.path.join(dataset_path, relative_audio_file)
                beats_file = os.path.join(beats_dir, audio_name+'.beats')
                dataset = AudioBeatsDatasetFromSong(audio_file, beats_file, precomputation_path,
                                                     duration, force_nb_samples)
                datasets.append(dataset)
    
            super().__init__(datasets)

### Dataset for evaluation (using full song feature)
class AudioBeatsEval(object):

    def __init__(self,
                 audio_file,
                 beats_file,
                 feature_file, 
                 ):
        self.audio_file       = audio_file
        self.beats_file       = beats_file
        self.feature_file     = feature_file 
        self.song_duration = librosa.get_duration(filename=self.audio_file) # total duration of the stretched wav
        if not os.path.exists(self.feature_file):
            self.precompute_feature()

    def get_wav(self):
        r"""Returns a numpy array of the audio section at the sampling rate
        determined by the `constants` module."""

        wav = librosa.load(self.audio_file,
                sr= global_sr, 
                )[0]
        return wav

    def get_beats(self):
        r"""Returns a numpy array of the beats in seconds.
        """
        if self.beats_file == None:
            print("Error: song {} 's beat file should be assigned".format(self.audio_file))
        else:
            all_beats = np.loadtxt(self.beats_file)

        return all_beats

    def precompute_feature(self):
        r"""Compute the mel-scaled spectrograms and store it in
        `self.spec_file`.
        """

        path = os.path.dirname(self.feature_file)
        if not os.path.exists(path):
            os.makedirs(path)
        features = utils.madmom_feature(self.get_wav())
        np.save(self.feature_file, features)

    def get_feature(self):
        r"""Returns the mel-scaled spectrogram (it must have been precomputed
        beforehand by calling `precompute_spec()` or `precompute()`)."""

        return np.load(self.feature_file)

    def precompute(self):
        r"""Precomputes the spectrogram, the onsets, and which onsets are beats.
        """
        self.precompute_feature()


    def get_data(self):
        r"""Returns the input feature all the beats (in seconds), audio file path.

        Returns:
            features (numpy array): filtered spectrograms calculated by Madmom API.
            beats (numpy array): list of beats (ground truth) in units seconds.
            audio_file (str): path of the audiofile.
        """

        features = self.get_feature()
        beats = self.get_beats()
        audio_file = self.audio_file
        
        return features, beats, audio_file

class EvalDataset(object):
    def __init__(self, audio_files):
        """ assuming data location aranged as:
        Dataset folder/
            audio_files.txt
            downbeats/ (folder for downbeat labelfiles)
            features/
                cutfeatures/
                fullfueatures/
            """
            
        dataset_path = os.path.dirname(audio_files)
        beats_dir = os.path.join(dataset_path, 'downbeats/')
        test_feature_dir = os.path.join(dataset_path, 'features', 'fullfeatures')
        
        datasets = []
        with open(audio_files) as f:
            for line in f.readlines():
                relative_audio_file = os.path.normpath(line.strip('\n'))
                audio_name = os.path.splitext(os.path.basename(relative_audio_file))[0]
                audio_file = os.path.join(dataset_path, relative_audio_file)
                beats_file = os.path.join(beats_dir, audio_name+'.beats')
                feature_file = os.path.join(test_feature_dir, audio_name+'.npy')
                
                dataset = AudioBeatsEval(audio_file, beats_file, feature_file)
                datasets.append(dataset)
        self.datasets = datasets

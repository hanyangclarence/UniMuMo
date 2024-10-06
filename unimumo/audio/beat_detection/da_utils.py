import librosa
import os
import numpy as np
from scipy.special import softmax
import torch

import unimumo.audio.beat_detection.drumaware_dataset as mmdataset
from madmom.audio.signal import SignalProcessor, FramedSignalProcessor
from madmom.audio.stft import ShortTimeFourierTransformProcessor
from madmom.audio.spectrogram import (
            FilteredSpectrogramProcessor, LogarithmicSpectrogramProcessor,
            SpectrogramDifferenceProcessor)
from madmom.processors import ParallelProcessor, SequentialProcessor


global_sr = 44100


def getExMixset(exMix_dataset_dirs, folderName = 'features', abname = 'train_dataset.ab',
                drum_abname = "osfq_qualified.ab",
                main_dir = './datasets/sourcesep_aug/', 
                NoDrum= True, OnlyDrum = True):
    if NoDrum:
        nodrumset = mmdataset.load_dataset(os.path.join(main_dir, exMix_dataset_dirs[0],
                                            "NoDrum", folderName, abname))

        for dataset in exMix_dataset_dirs[1:]:
            ssaug_dir = os.path.join(main_dir, dataset)
            nodrumset+= mmdataset.load_dataset(os.path.join(ssaug_dir, "NoDrum", 
                                                            folderName, abname))
    if OnlyDrum:
        onlydrumset = mmdataset.load_dataset(os.path.join(main_dir, exMix_dataset_dirs[0], 
                                            "OnlyDrum", folderName, drum_abname))
        for dataset in exMix_dataset_dirs[1:]:
            ssaug_dir = os.path.join(main_dir, dataset)
            onlydrumset+= mmdataset.load_dataset(os.path.join(ssaug_dir, "OnlyDrum", 
                                                              folderName, drum_abname))
    if NoDrum and OnlyDrum:
        print("===using both NoDrum and OnlyDrum===")
        return nodrumset + onlydrumset
    elif NoDrum and not OnlyDrum:
        print("===using only NoDrum===")
        return nodrumset
    elif OnlyDrum and not NoDrum:
        print("===using only OnlyDrum===")
        return onlydrumset
    else:
        print("======Something is Wrong in Your getExMixset settings!!!======")

def getMixset(mix_dataset_dirs, folderName ='features', abname = 'train_dataset.ab',
              main_dir = './datasets/original/'):
    mixset = mmdataset.load_dataset(os.path.join(main_dir, mix_dataset_dirs[0], 
                                                 folderName, abname ))

    for dataset in mix_dataset_dirs[1:]:
        mixset_dir = os.path.join(main_dir, dataset, folderName)
        mixset += mmdataset.load_dataset(os.path.join(mixset_dir, abname ))
    return mixset

def get_wav(audio_file_path):
    wav = librosa.load(audio_file_path,
                sr= global_sr, )[0]
    return wav
    
def get_beats(beats_file_txt):

    all_beats = np.loadtxt(beats_file_txt)

    return all_beats

def get_feature(audio_file_path):

    features = madmom_feature(get_wav(audio_file_path))
    return features

### calculating filtered spectrograms and first order derivative using Madmom API
def madmom_feature(wav):
    """ returns the madmom features mentioned in the paper"""
    sig = SignalProcessor(num_channels=1, sample_rate=global_sr )
    multi = ParallelProcessor([])
    frame_sizes = [1024, 2048, 4096]
    num_bands = [3, 6, 12]
    for frame_size, num_bands in zip(frame_sizes, num_bands):
        frames = FramedSignalProcessor(frame_size=frame_size, fps=100)
        stft = ShortTimeFourierTransformProcessor()  # caching FFT window
        filt = FilteredSpectrogramProcessor(
            num_bands=num_bands, fmin=30, fmax=17000, norm_filters=True)
        spec = LogarithmicSpectrogramProcessor(mul=1, add=1)
        diff = SpectrogramDifferenceProcessor(
            diff_ratio=0.5, positive_diffs=True, stack_diffs=np.hstack)
        # process each frame size with spec and diff sequentially
        multi.append(SequentialProcessor((frames, stft, filt, spec, diff)))
    # stack the features and processes everything sequentially
    pre_processor = SequentialProcessor((sig, multi, np.hstack))
    feature = pre_processor.process( wav)
    return feature

def df2eval_dictlist(df, withMadmom = False):
    if withMadmom:
        eval_dictlist = [
                        {
            'model_type': 'madmom_api', # should allow using api for comparison 
            'model_dir': None, # only allow None when using madom_api
            'model_simpname': 'Madmom', 
            'n_tempi':  60, 
            'transition_lambda': 100, 
            'observation_lambda': 16, 
            'threshold': 0.05,
            'model_setting': 'nan',
            
        }, 
        ]
    else:
        eval_dictlist = []
    for model_ind in range(len(df)):
#        break
        model_dict={
                'model_type': df['model_type'].iloc[model_ind],
                'model_dir': df['model_dir'].iloc[model_ind], 
                'model_simpname': df['model_simpname'].iloc[model_ind], 
                'model_setting': df['model_setting'].iloc[model_ind],
                'n_tempi': df['n_tempi'].iloc[model_ind], 
                'transition_lambda': df['transition_lambda'].iloc[model_ind], 
                'observation_lambda':df['observation_lambda'].iloc[model_ind], 
                'threshold': df['threshold'].iloc[model_ind],

                }
        eval_dictlist.append(model_dict)
    return eval_dictlist

def prediction_conversion(prediction):
    if len(prediction.shape) == 2:
        prediction = prediction.unsqueeze(0)
    pred_arr = prediction.detach().cpu().numpy()
    pred_acti = softmax(pred_arr, axis = 2)
    pred_acti = pred_acti.squeeze()
    
    model_activation = np.zeros((pred_acti.shape[0], 2))
    model_activation[:, 0] = pred_acti[:, 2] # beat class
    model_activation[:, 1] = pred_acti[:, 1] # downbeat class
    return model_activation

def get_dlm_activation(rnn, device, np_2dfeature):
    """ get deep learning model activations"""
    input_feature = torch.tensor(np_2dfeature[np.newaxis, :, :]).float().to(device)
    rnn.eval()
    rnn.to(device)
    
    with torch.no_grad():
        activation = rnn(input_feature)
    
    ### For DA models 
    if type(activation)==tuple and len(activation) ==6:
        beat_fused, beat_mix, beat_nodrum, beat_drum, x_nodrum_hat, x_drum_hat = activation
        fuser_activation = prediction_conversion(beat_fused)
        mix_activation = prediction_conversion(beat_mix)
        nodrum_activation = prediction_conversion(beat_nodrum)
        drum_activation = prediction_conversion(beat_drum)
        model_activation = [fuser_activation, mix_activation, nodrum_activation, drum_activation]
        
        return model_activation
    
    else:
        beat_fused = activation.clone().detach()
    
        fuser_activation = prediction_conversion(beat_fused)
        
        return fuser_activation


### Functions for saving best models
### below functions were modified from source code: https://github.com/sigsep/open-unmix-pytorch/blob/master/openunmix/utils.py
def save_checkpoint(
    state, is_best, path, target):
    # save full checkpoint including optimizer
    torch.save(
        state,
        os.path.join(path, target + '.chkpnt')
    )
    if is_best:
        # save just the weights
        torch.save(
            state['state_dict'],
            os.path.join(path, target + '.pth')
        )


class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=10, best_loss = None):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = best_loss
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta)

        if patience == 0:
            self.is_better = lambda a, b: True

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if np.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if mode == 'min':
            self.is_better = lambda a, best: a < best - min_delta
        if mode == 'max':
            self.is_better = lambda a, best: a > best + min_delta











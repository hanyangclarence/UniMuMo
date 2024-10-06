# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 15:16:17 2020

 
@author: CITI
"""
#%%
from torch.nn import LSTM, Linear, BatchNorm1d, Parameter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

class RNNDownBeatProc(nn.Module):
    def __init__(
        self, 
        feature_size = 314, 
        blstm_hidden_size = 25,
        nb_layers = 3, 
        out_features = 3, 
        dropout = 0, 
        two_stage_feature_size = 25, # size of output of feature_fc layer before prediction fc layer
        ):
        """
        input: (nb_frames, nb_samples, feature_size)
        output: (nb_frames, nb_samples, 3)
        3: beat, downbeat, non-beat activations """
        
        super(RNNDownBeatProc, self).__init__()
        
        self.two_stage_feature_size = two_stage_feature_size
        self.lstm = LSTM(
            input_size = feature_size, 
            hidden_size = blstm_hidden_size, 
            num_layers = nb_layers, 
            bidirectional = True, 
            batch_first = True, 
            dropout = 0)# not sure
        
        if not two_stage_feature_size:
            self.fc1 = Linear(
                    in_features = blstm_hidden_size*2, 
                    out_features = 3, 
                    bias = True) # 2.2.2 in paper mentioned bias
        else:
            self.feature_fc = Linear(
                    in_features = blstm_hidden_size*2, 
                    out_features = two_stage_feature_size, 
                    bias = True)
            self.fc1 = Linear(in_features = two_stage_feature_size, 
                              out_features = out_features, 
                              bias = True)
        
        self.reset_params()
    @staticmethod
    def weight_init(m):
        classname = m.__class__.__name__
        if classname=="Linear":
            init.uniform_( m.weight,  a = -0.1, b = 0.1)
            init.uniform_( m.bias, a = -0.1, b = 0.1)
    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)
    
    def forward(self, x):
        x = self.lstm(x)
        if not self.two_stage_feature_size:
            x = self.fc1(x[0])
            return x
        else:
            feature = self.feature_fc(x[0])
            x = self.fc1(feature)

            return x, feature

class BeatOpenUnmix(nn.Module):
    def __init__(
        self,
        input_size = 314,
        hidden_size=128, 
        output_size = 314,
        nb_layers=3,
        input_mean=None,
        input_scale=None,
        unidirectional=False,
    ):
        """
        Input: (nb_samples, nb_channels, nb_timesteps)
            or (nb_frames, nb_samples, nb_channels, nb_bins)
        Output: Power/Mag Spectrogram
                (nb_frames, nb_samples, nb_channels, nb_bins)
        """

        super(BeatOpenUnmix, self).__init__()

        self.nb_output_bins = output_size
        self.hidden_size = hidden_size

        self.input_size = input_size
        self.fc1 = Linear(
            self.input_size, hidden_size,
            bias=False
        )

        self.bn1 = BatchNorm1d(hidden_size)

        if unidirectional:
            lstm_hidden_size = hidden_size
        else:
            lstm_hidden_size = hidden_size // 2 
            # for bidirection, each direction would output lstm_hidden_size's output, 
            # therefore, should be divided by 2

        self.lstm = LSTM(
            input_size=hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=nb_layers,
            bidirectional=not unidirectional,
            batch_first=True,
            dropout=0.2,
        )

        self.fc2 = Linear(
            in_features=hidden_size*2, # 1 hidden_size from lstm output, 1 hidden_size from skip connection
            out_features=hidden_size,
            bias=False
        )

        self.bn2 = BatchNorm1d(hidden_size)

        self.fc3 = Linear(
            in_features=hidden_size,
            out_features=self.nb_output_bins,
            bias=False
        )

        self.bn3 = BatchNorm1d(self.nb_output_bins)

        if input_mean is not None:
            input_mean = torch.from_numpy(
                -input_mean[:self.input_size]
            ).float()
        else:
            input_mean = torch.zeros(self.input_size)

        if input_scale is not None:
            input_scale = torch.from_numpy(
                1.0/input_scale[:self.input_size]
            ).float()
        else:
            input_scale = torch.ones(self.input_size)

        self.input_mean = Parameter(input_mean)
        self.input_scale = Parameter(input_scale)

        self.output_scale = Parameter(
            torch.ones(self.nb_output_bins).float()
        )
        self.output_mean = Parameter(
            torch.ones(self.nb_output_bins).float()
        )

    def forward(self, x):

        nb_samples, nb_frames, nb_bins = x.data.shape

        mix = x.detach().clone()
        x += self.input_mean
        x *= self.input_scale

        x = self.fc1(x.reshape(-1, self.input_size))
        # normalize every instance in a batch
        x = self.bn1(x)
        x = x.reshape(nb_samples, nb_frames, self.hidden_size)
        # squash range ot [-1, 1]
        x = torch.tanh(x)

        # apply 3-layers of stacked LSTM
        lstm_out = self.lstm(x)

        # lstm skip connection
        x = torch.cat([x, lstm_out[0]], -1)

        # first dense stage + batch norm
        x = self.fc2(x.reshape(-1, x.shape[-1]))
        x = self.bn2(x)

        x = F.relu(x)

        # second dense stage + layer norm
        x = self.fc3(x)
        x = self.bn3(x)

        # reshape back to original dim
        x = x.reshape(nb_samples, nb_frames, self.nb_output_bins)

        # apply output scaling
        x *= self.output_scale
        x += self.output_mean

        # since our output is non-negative, we can apply RELU
        x = F.relu(x) * mix

        return x


        
class DrumAwareBeatTracker(nn.Module):
    def __init__(
            self, 
            OU_chkpnt = None, 
            DrumOU_chkpnt = None, 
            DrumBeat_chkpnt = None, 
            NDrumBeat_chkpnt = None, 
            MixBeat_chkpnt = None, 
            FuserBeat_chkpnt = None,
            fixed_DrumOU = True,
            fixed_drum = True, 
            fixed_OU = True,
            fixed_mix = True, 
            fixed_nodrum = True,
            fixed_fuser = False,
            
            drum_nb_layers =3, 
            drum_blstm_hidden_size = 25, 
            drum_out_features = 3, 
            drum_dropout = 0, 
            drum_2stage_fsize = 0,
            
            mix_nb_layers = 3, 
            mix_blstm_hidden_size = 25,
            mix_out_features = 3, 
            mix_dropout = 0, 
            mix_2stage_fsize = 0,
            
            nodrum_nb_layers = 3, 
            nodrum_blstm_hidden_size = 25,
            nodrum_out_features = 3,
            nodrum_dropout = 0, 
            nodrum_2stage_fsize = 0, 
            
            fuser_nb_layers = 3, 
            fuser_blstm_hidden_size = 25,
            fuser_out_features = 3, 
            fuser_dropout = 0, 
            fuser_2stage_fsize = 0,
            
            beatou_nb_layers =3,
            drumou_nb_layers = 3, 
            feature_size = 314):
        super(DrumAwareBeatTracker, self).__init__()
            
        self.fixed_OU = fixed_OU
        self.fixed_mix = fixed_mix
        self.fixed_nodrum = fixed_nodrum
        self.fixed_fuser = fixed_fuser
        self.fixed_DrumOU = fixed_DrumOU 
        self.fixed_drum = fixed_drum
        print("======Model Init======")
        print("Fixed OU:{}, DrumOU:{}, Mix:{}, NoDrum:{}, Drum:{}, Fuser:{}".format(self.fixed_OU, 
              self.fixed_DrumOU, self.fixed_mix, 
              self.fixed_nodrum, self.fixed_drum, self.fixed_fuser))
        self.BeatOU = BeatOpenUnmix(input_size = feature_size, nb_layers = beatou_nb_layers)
        if OU_chkpnt:
            ou_state = torch.load(OU_chkpnt)
            self.BeatOU.load_state_dict(ou_state)
            print("====Loaded trained BeatOU!====")
        
        self.DrumOU = BeatOpenUnmix(input_size = feature_size, nb_layers = drumou_nb_layers)
        if DrumOU_chkpnt: 
            drum_ou_state = torch.load(DrumOU_chkpnt)
            self.DrumOU.load_state_dict(drum_ou_state)
            print("====Loaded trained DrumOU!====") 
            
        self.drumBeat = RNNDownBeatProc(feature_size, drum_blstm_hidden_size, drum_nb_layers, 
                                        out_features = drum_out_features, dropout = drum_dropout, 
                                        two_stage_feature_size= drum_2stage_fsize)
        if DrumBeat_chkpnt:
            drum_beat_state = torch.load(DrumBeat_chkpnt)
            self.drumBeat.load_state_dict(drum_beat_state)
            print("====Load trained DrumBeat!====")
        
        self.mixBeat = RNNDownBeatProc(feature_size, mix_blstm_hidden_size, mix_nb_layers, 
                                       out_features = mix_out_features, dropout = mix_dropout, 
                                       two_stage_feature_size= mix_2stage_fsize)
        if MixBeat_chkpnt:
            mix_beat_state = torch.load(MixBeat_chkpnt)
            self.mixBeat.load_state_dict(mix_beat_state)
            print("====Loaded trained MixBeat!====")
            
        self.nodrumBeat = RNNDownBeatProc(feature_size, nodrum_blstm_hidden_size, nodrum_nb_layers, 
                                          out_features = nodrum_out_features, dropout = nodrum_dropout, 
                                          two_stage_feature_size= nodrum_2stage_fsize)
        if NDrumBeat_chkpnt:
            ndrum_beat_state = torch.load(NDrumBeat_chkpnt)
            self.nodrumBeat.load_state_dict(ndrum_beat_state)
            print("====Loaded trained NDrumBeat!====")
            
        self.fuserBeat = RNNDownBeatProc(nodrum_out_features + drum_out_features + mix_out_features +\
                                         nodrum_2stage_fsize+ drum_2stage_fsize + mix_2stage_fsize, 
                                         fuser_blstm_hidden_size, fuser_nb_layers, 
                                         out_features = fuser_out_features, dropout = fuser_dropout, 
                                         two_stage_feature_size= fuser_2stage_fsize)
        if FuserBeat_chkpnt:
            fuser_beat_state = torch.load(FuserBeat_chkpnt)
            self.fuserBeat.load_state_dict(fuser_beat_state)
            print("====Loaded trained FuserBeat!====")
            
    def forward(self, x):
        mix = x.detach().clone()
        if self.fixed_OU:
            self.BeatOU.eval()
            with torch.no_grad():
                x_nodrum = self.BeatOU(mix)
        else:
            x_nodrum = self.BeatOU(mix)
        
        if self.fixed_DrumOU:
            self.DrumOU.eval()
            with torch.no_grad():
                x_drum = self.DrumOU(x)
        else:
            x_drum = self.DrumOU(x)
        
        if self.fixed_drum:
            self.drumBeat.eval()
            with torch.no_grad():
                beat_drum, feature_drum = self.drumBeat(x_drum.detach())
        else:
            beat_drum, feature_drum = self.drumBeat(x_drum.detach())
                
        
        if self.fixed_nodrum:
            self.nodrumBeat.eval()
            with torch.no_grad():
                beat_nodrum, feature_nodrum = self.nodrumBeat(x_nodrum.detach())
        else:
            beat_nodrum, feature_nodrum = self.nodrumBeat(x_nodrum.detach())
            
        if self.fixed_mix:
            self.mixBeat.eval()
            with torch.no_grad():
                beat_mix, feature_mix = self.mixBeat(x)
        else:
            beat_mix, feature_mix = self.mixBeat(x)
        combined_pred = torch.cat([feature_mix.detach(), feature_nodrum.detach(), feature_drum.detach(), 
                                   beat_mix.detach(), beat_nodrum.detach(), beat_drum.detach()], dim = -1)
        
        if self.fixed_fuser:
            self.fuserBeat.eval()
            with torch.no_grad():
                beat_fused = self.fuserBeat(combined_pred)
        else:
            beat_fused = self.fuserBeat(combined_pred)
        return beat_fused, beat_mix, beat_nodrum, beat_drum, x_nodrum, x_drum

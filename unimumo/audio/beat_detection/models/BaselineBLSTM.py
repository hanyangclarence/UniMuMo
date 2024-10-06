# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 15:50:07 2020

@author: CITI
"""

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
        ):
        """
        input: (nb_frames, nb_samples, feature_size)
        output: (nb_frames, nb_samples, 3)
        3: beat, downbeat, non-beat activations """
        
        super(RNNDownBeatProc, self).__init__()
        
        self.lstm = LSTM(
            input_size = feature_size, 
            hidden_size = blstm_hidden_size, 
            num_layers = nb_layers, 
            bidirectional = True, 
            batch_first = True, 
            dropout = 0)# not sure
        
        self.fc1 = Linear(
                in_features = blstm_hidden_size*2, 
                out_features = 3, # beat, downbeat, non-beat activations
                bias = True) # 2.2.2 in paper mentioned bias
        
#        self.activation = nn.Softmax(dim=1)
#        self.activation = nn.LogSoftmax(dim = 1)
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
        x = self.fc1(x[0])
#        x = self.activation(x)
        return x
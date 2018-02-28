# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import division

import numpy as np
import torch

from torch import nn
from torch.nn import functional as F

import melfilters
import utils


class TDFbanks(nn.Module):
    def __init__(self,
                 mode,
                 nfilters,
                 samplerate=16000,
                 wlen=25,
                 wstride=10,
                 compression='log',
                 preemp=False,
                 mvn=False):
        super(TDFbanks, self).__init__()
        window_size = samplerate * wlen // 1000 + 1
        window_stride = samplerate * wstride // 1000
        padding_size = (window_size - 1) // 2
        self.preemp = None
        if preemp:
            self.preemp = nn.Conv1d(1, 1, 2, 1, padding=1, groups=1, bias=False)
        self.complex_conv = nn.Conv1d(1, 2 * nfilters, window_size, 1,
            padding=padding_size, groups=1, bias=False)
        self.modulus = nn.LPPool1d(2, 2, stride=2)
        self.lowpass = nn.Conv1d(nfilters, nfilters, window_size, window_stride,
            padding=0, groups=nfilters, bias=False)
        if mode == 'Fixed':
            for param in self.parameters():
                param.requires_grad = False
        elif mode == 'learnfbanks':
            if preemp:
                self.preemp.weight.requires_grad = False
            self.lowpass.weight.requires_grad = False
        if mvn:
            self.instancenorm = nn.InstanceNorm1d(nfilters, momentum=1)
        self.nfilters = nfilters
        self.fs = samplerate
        self.wlen = wlen
        self.wstride = wstride
        self.compression = compression
        self.mvn = mvn


    def initialize(self,
                   min_freq=0,
                   max_freq=8000,
                   nfft=512,
                   window_type='hanning',
                   normalize_energy=False,
                   alpha=0.97):
        # Initialize preemphasis
        if self.preemp:
            self.preemp.weight.data[0][0][0] = -alpha
            self.preemp.weight.data[0][0][1] = 1
        # Initialize complex convolution
        self.complex_init = melfilters.Gabor(self.nfilters,
                                             min_freq,
                                             max_freq,
                                             self.fs,
                                             self.wlen,
                                             self.wstride,
                                             nfft,
                                             normalize_energy)
        for idx, gabor in enumerate(self.complex_init.gaborfilters):
            self.complex_conv.weight.data[2*idx][0].copy_(
                torch.from_numpy(np.real(gabor)))
            self.complex_conv.weight.data[2*idx + 1][0].copy_(
                torch.from_numpy(np.imag(gabor)))
        # Initialize lowpass
        self.lowpass_init = utils.window(window_type,
                                         (self.fs * self.wlen)//1000 + 1)
        for idx in range(self.nfilters):
            self.lowpass.weight.data[idx][0].copy_(
                torch.from_numpy(self.lowpass_init))

    def forward(self, x):
        # Reshape waveform to format (1,1,seq_length)
        x = x.view(1, 1, -1)
        # Preemphasis
        if self.preemp:
            x = self.preemp(x)
        # Complex convolution
        x = self.complex_conv(x)
        # Modulus operator
        x = x.transpose(1, 2)
        x = self.modulus(x)
        x = x.transpose(1, 2)
        # Square
        x = x.pow(2)
        x = self.lowpass(x)
        x = x.abs()
        x = x + 1
        if self.compression == 'log':
            x = x.log()
        # The dimension of x is 1, n_channels, seq_length
        if self.mvn:
            x = self.instancenorm(x)
        return x

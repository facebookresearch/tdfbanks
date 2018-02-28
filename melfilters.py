# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import division
import numpy as np


class Gabor(object):
    def __init__(self,
                 nfilters=40,
                 min_freq=0,
                 max_freq=8000,
                 fs=16000,
                 wlen=25,
                 wstride=10,
                 nfft=512,
                 normalize_energy=False):
            if not nfilters > 0:
                raise(Exception,
                'Number of filters must be positive, not {0:%d}'.format(nfilters))
            if max_freq > fs // 2:
                raise(Exception,
                'Upper frequency %f exceeds Nyquist %f' % (max_freq, fs // 2))
            self.nfilters = nfilters
            self.min_freq = min_freq
            self.max_freq = max_freq
            self.fs = fs
            self.wlen = wlen
            self.wstride = wstride
            self.nfft = nfft
            self.normalize_energy = normalize_energy
            self._build_mels()
            self._build_gabors()

    def _hz2mel(self, f):
        # Converts a frequency in hertz to mel
        return 2595 * np.log10(1+f/700)

    def _mel2hz(self, m):
        # Converts a frequency in mel to hertz
        return 700 * (np.power(10, m/2595) - 1)

    def _gabor_wavelet(self, eta, sigma):
        T = self.wlen * self.fs / 1000
        # Returns a gabor wavelet on a window of size T

        def gabor_function(t):
            return (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(1j * eta * t) * np.exp(-t**2/(2 * sigma**2))
        return np.asarray([gabor_function(t) for t in np.arange(-T/2,T/2 + 1)])

    def _gabor_params_from_mel(self, mel_filter):
        # Parameters in radians
        coeff = np.sqrt(2*np.log(2))*self.nfft
        mel_filter = np.sqrt(mel_filter)
        center_frequency = np.argmax(mel_filter)
        peak = mel_filter[center_frequency]
        half_magnitude = peak/2.0
        spread = np.where(mel_filter >= half_magnitude)[0]
        width = max(spread[-1] - spread[0],1)
        return center_frequency*2*np.pi/self.nfft, coeff/(np.pi*width)

    def _melfilter_energy(self, mel_filter):
        # Computes the energy of a mel-filter (area under the magnitude spectrum)
        height = max(mel_filter)
        hz_spread = (len(np.where(mel_filter > 0)[0])+2)*2*np.pi/self.nfft
        return 0.5 * height * hz_spread

    def _build_mels(self):
        # build mel filter matrix
        self.melfilters = [np.zeros(self.nfft//2 + 1) for i in range(self.nfilters)]
        dfreq = self.fs / self.nfft

        melmax = self._hz2mel(self.max_freq)
        melmin = self._hz2mel(self.min_freq)
        dmelbw = (melmax - melmin) / (self.nfilters + 1)
        # filter edges in hz
        filt_edge = self._mel2hz(melmin + dmelbw *
                                 np.arange(self.nfilters + 2, dtype='d'))
        self.filt_edge = filt_edge
        for filter_idx in range(0, self.nfilters):
            # Filter triangles in dft points
            leftfr = min(round(filt_edge[filter_idx] / dfreq), self.nfft//2)
            centerfr = min(round(filt_edge[filter_idx + 1] / dfreq), self.nfft//2)
            rightfr = min(round(filt_edge[filter_idx + 2] / dfreq), self.nfft//2)
            height = 1
            if centerfr != leftfr:
                leftslope = height / (centerfr - leftfr)
            else:
                leftslope = 0
            freq = leftfr + 1
            while freq < centerfr:
                self.melfilters[filter_idx][int(freq)] = (freq - leftfr) * leftslope
                freq += 1
            if freq == centerfr:
                self.melfilters[filter_idx][int(freq)] = height
                freq += 1
            if centerfr != rightfr:
                rightslope = height / (centerfr - rightfr)
            while freq < rightfr:
                self.melfilters[filter_idx][int(freq)] = (freq - rightfr) * rightslope
                freq += 1
            if self.normalize_energy:
                energy = self._melfilter_energy(self.melfilters[filter_idx])
                self.melfilters[filter_idx] /= energy

    def _build_gabors(self):
        self.gaborfilters = []
        self.sigmas = []
        self.center_frequencies = []
        for mel_filter in self.melfilters:
            center_frequency, sigma = self._gabor_params_from_mel(mel_filter)
            self.sigmas.append(sigma)
            self.center_frequencies.append(center_frequency)
            gabor_filter = self._gabor_wavelet(center_frequency, sigma)
            # Renormalize the gabor wavelets
            gabor_filter = gabor_filter * np.sqrt(self._melfilter_energy(mel_filter)*2*np.sqrt(np.pi)*sigma)
            self.gaborfilters.append(gabor_filter)

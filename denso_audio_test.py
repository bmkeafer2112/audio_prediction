# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 09:32:59 2023

@author: bmkea
"""
import numpy as np
import librosa
import torch
import soundfile
import matplotlib.pyplot as plt
from Denso_Audio import AudioUtil

audio = AudioUtil()

y, sr = librosa.load(librosa.ex('trumpet'))

file= audio.open('denso_added_tone.wav')

new_channel = audio.rechannel(file, 1)

resample = audio.resample(new_channel, 44100)

new_length = audio.pad_trunc(resample, 4000)

new_time = audio.time_shift(new_length, 2)

spect = audio.spectro_gram(new_time)

aug_spect = audio.spectro_augment(spect)

#time_mask = audio.time_masking(resample)

sig00, sr00 = file
sig01, sr01 = new_channel
sig02, sr02 = resample
sig03, sr03 = new_length
sig04, sr04 = new_time
sig05, sr05 = spect
sig06, sr06 = aug_spect
#sig07, sr07 = time_mask


audio.plot_waveform(sig00, sr00)
audio.plot_mel_spectogram(sig00, sr00)

audio.plot_waveform(sig01, sr01)
audio.plot_mel_spectogram(sig01, sr01)

audio.plot_waveform(sig02, sr02)
audio.plot_mel_spectogram(sig02, sr02)

audio.plot_waveform(sig03, sr03)
audio.plot_mel_spectogram(sig03, sr03)

audio.plot_waveform(sig04, sr04)
audio.plot_mel_spectogram(sig04, sr04)

#audio.plot_waveform(sig07, sr07)
#audio.plot_mel_spectogram(sig07, sr07)




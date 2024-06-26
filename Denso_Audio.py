# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 12:02:34 2023

@author: bmkea
"""

import numpy as np
import librosa
import math, random
import os
import torch
import torchaudio
from torchaudio import transforms
from IPython.display import Audio
import matplotlib.pyplot as plt

class AudioUtil():
# ----------------------------
# Load an audio file. Return the signal as a tensor and the sample rate
# ----------------------------
    @staticmethod
    def open(audio_file):
        sig, sr = torchaudio.load(audio_file)
        return (sig, sr)
    
# ----------------------------
# Convert the given audio to the desired number of channels
# ----------------------------
    @staticmethod
    def rechannel(aud, new_channel):
      sig, sr = aud
    
      if (sig.shape[0] == new_channel):
        # Nothing to do
        return aud
    
      if (new_channel == 1):
        # Convert from stereo to mono by selecting only the first channel
        resig = sig[:1, :]
      else:
        # Convert from mono to stereo by duplicating the first channel
        resig = torch.cat([sig, sig])
    
      return ((resig, sr))
  
# ----------------------------
# Since Resample applies to a single channel, we resample one channel at a time
# ----------------------------
    @staticmethod
    def resample(aud, newsr):
      sig, sr = aud
    
      if (sr == newsr):
        # Nothing to do
        return aud
    
      num_channels = sig.shape[0]
      # Resample first channel
      resig = torchaudio.transforms.Resample(sr, newsr)(sig[:1,:])
      if (num_channels > 1):
        # Resample the second channel and merge both channels
        retwo = torchaudio.transforms.Resample(sr, newsr)(sig[1:,:])
        resig = torch.cat([resig, retwo])
    
      return ((resig, newsr))
  
# ----------------------------
# Pad (or truncate) the signal to a fixed length 'max_ms' in milliseconds
# ----------------------------
    @staticmethod
    def pad_trunc(aud, max_ms):
      sig, sr = aud
      num_rows, sig_len = sig.shape
      max_len = sr//1000 * max_ms
    
      if (sig_len > max_len):
        # Truncate the signal to the given length
        sig = sig[:,:max_len]
    
      elif (sig_len < max_len):
        # Length of padding to add at the beginning and end of the signal
        pad_begin_len = random.randint(0, max_len - sig_len)
        pad_end_len = max_len - sig_len - pad_begin_len
    
        # Pad with 0s
        pad_begin = torch.zeros((num_rows, pad_begin_len))
        pad_end = torch.zeros((num_rows, pad_end_len))
    
        sig = torch.cat((pad_begin, sig, pad_end), 1)
        
      return (sig, sr)
    
# ----------------------------
# Shifts the signal to the left or right by some percent. Values at the end
# are 'wrapped around' to the start of the transformed signal.
# ----------------------------
    @staticmethod
    def time_shift(aud, shift_limit):
      sig,sr = aud
      _, sig_len = sig.shape
      shift_amt = int(random.random() * shift_limit * sig_len)
      return (sig.roll(shift_amt), sr)

# ----------------------------
# Generate a Spectrogram
# ----------------------------
    @staticmethod
    def spectro_gram(aud, n_mels=64, n_fft=1024, hop_len=None):
      sig,sr = aud
      top_db = 80
    
      # spec has shape [channel, n_mels, time], where channel is mono, stereo etc
      spec = transforms.MelSpectrogram(sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(sig)
    
      # Convert to decibels
      spec = transforms.AmplitudeToDB(top_db=top_db)(spec)
      return (spec, sr)
  
# ----------------------------
# Augment the Spectrogram by masking out some sections of it in both the frequency
# dimension (ie. horizontal bars) and the time dimension (vertical bars) to prevent
# overfitting and to help the model generalise better. The masked sections are
# replaced with the mean value.
# ----------------------------
    @staticmethod
    def spectro_augment(spec, max_mask_pct=0.1, n_freq_masks=1, n_time_masks=1):
      aug_spec, sr = spec
      _, n_mels, n_steps = aug_spec.shape
      mask_value = aug_spec.mean()
    
      freq_mask_param = max_mask_pct * n_mels
      for _ in range(n_freq_masks):
        aug_spec = transforms.FrequencyMasking(freq_mask_param)(aug_spec, mask_value)
    
      time_mask_param = max_mask_pct * n_steps
      for _ in range(n_time_masks):
        aug_spec = transforms.TimeMasking(time_mask_param)(aug_spec, mask_value)
    
      return (aug_spec, sr)
  
    @staticmethod
    def plot_waveform(waveform, sample_rate, title="Waveform", xlim=None, ylim=None):
      waveform = waveform.numpy()
    
      num_channels, num_frames = waveform.shape
      time_axis = torch.arange(0, num_frames) / sample_rate
    
      figure, axes = plt.subplots(num_channels, 1)
      if num_channels == 1:
        axes = [axes]
      for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
          axes[c].set_ylabel(f'Channel {c+1}')
        if xlim:
          axes[c].set_xlim(xlim)
        if ylim:
          axes[c].set_ylim(ylim)
      figure.suptitle(title)
      plt.show(block=False)
     
    @staticmethod  
    def plot_spectrogram(spec, title=None, ylabel='freq_bin', aspect='auto', xmax=None):
      fig, axs = plt.subplots(1, 1)
      axs.set_title(title or 'Spectrogram (db)')
      axs.set_ylabel(ylabel)
      axs.set_xlabel('frame')
      im = axs.imshow(librosa.power_to_db(spec), origin='lower', aspect=aspect)
      if xmax:
        axs.set_xlim((0, xmax))
      fig.colorbar(im, ax=axs)
      plt.show(block=False)
      
    @staticmethod
    def plot_mel_spectogram(waveform, sample_rate, title="Mel-Spectogram Display"):
        fig, ax = plt.subplots()
        waveform = np.array(waveform)
        waveform = np.transpose(waveform)
        length = len(waveform)
        waveform = np.reshape(waveform, (length,))     
        M = librosa.feature.melspectrogram(y=waveform, sr=sample_rate)
        M_db = librosa.power_to_db(M, ref=np.max)
        img = librosa.display.specshow(M_db, y_axis='mel', x_axis='time', ax=ax)
        ax.set(title='Mel spectrogram display')
        fig.colorbar(img, ax=ax, format="%+2.f dB")
        
    @staticmethod
    def time_masking(aud):
        waveform, sr = aud
        masking = transforms.TimeMasking(time_mask_param=80)
        spec = masking(waveform)
        return (spec, sr)
  


#audio = AudioUtil()
#sig, sr = audio.open(r"C:\Users\bmkea\Documents\Denso_Test_cell\Python Scripts\Audio_Prediction\Audio_Files\Denso_4_Seconds\denso_4sec_1.wav")

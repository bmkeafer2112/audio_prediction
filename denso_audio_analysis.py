# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 12:02:34 2023

@author: bmkea
"""

import math, random
import torch
import torchaudio
from torchaudio import transforms
from IPython.display import Audio

class AudioUtil():
  # ----------------------------
  # Load an audio file. Return the signal as a tensor and the sample rate
  # ----------------------------
  @staticmethod
  def open(audio_file):
    sig, sr = torchaudio.load(audio_file)
    return (sig, sr)

audio = AudioUtil()
sig, sr = audio.open(r"C:\Users\bmkea\Documents\Denso_Test_cell\Python Scripts\Audio_Prediction\Audio_Files\Denso_4_Seconds\denso_4sec_1.wav")

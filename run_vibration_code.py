# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 09:45:44 2024

@author: bmkea
"""

from vibration import vibration

#Record Vibration data and Display Spectogram
vibe = vibration()
records, dic, frame = vibe.record_vibration(chunk = 1024, channels = 8, rate = 48000, record_time = 21)

vibe.build_melspectogram(wave_files = records)

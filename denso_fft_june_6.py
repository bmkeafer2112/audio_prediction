# -*- coding: utf-8 -*-
"""
Created on Tue May 30 09:30:03 2023

@author: Canon
"""

import os
from scipy.fft import fft, fftfreq
from scipy.fft import rfft, rfftfreq
from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np
from statistics import mean, median

binFreqFlag = True 
semilogFlag = True

# Robot Moving Audio File Name
filename = "denso_1sec_1.wav"

# FFT Settings
low_hz = 100 #Lowest is 0
high_hz = 20000 #Highest is 96001

low_threshold_samples = 100000
low_threshold_ratio = 0.1

# Audio without Robots Moving
_ , robotoffdata = wavfile.read("C:\\Users\Canon\Desktop\denso_1sec_3.wav")

# Audio with Robots Moving
wav_fname = os.path.join("C:\\Users\Canon\Desktop", filename)
samplerate, data = wavfile.read(wav_fname)

# Time Length of Audio of Robots Moving
length = data.shape[0] / samplerate
time = np.linspace(0.,length, data.shape[0])

# FFT Data -- Background Noise
noisex = rfftfreq(robotoffdata.shape[0], 1/samplerate)
noisey = rfft(robotoffdata)

# FFT Data -- Robot Moving
fftx = rfftfreq(data.shape[0], 1/samplerate)
ffty = rfft(data)


# Zoom Settings for FFT Graph
f_ratio_low = low_hz / samplerate
f_ratio_high = high_hz / samplerate
lowindex = int(len(data)*f_ratio_low)
highindex = int(len(data)*f_ratio_high)

def binFreq(freq, magnitude, starting_value):
    currentfreq = starting_value
    currentdata = []
    newfreq = []
    newdata = []
    for i in range(len(freq)): 
        roundedfreq = round(freq[i],0)
        
        if  currentfreq != roundedfreq:
            newfreq.append(currentfreq)
            
            if len(currentdata) != 0:
                newdata.append(mean(currentdata))
                currentdata = []
            else:
                newdata.append(0)

            currentfreq = roundedfreq    
            currentdata.append(magnitude[i])
            
        elif i == len(freq) - 1:
            if magnitude[i] > low_threshold_samples:
                currentdata.append(magnitude[i])
            newfreq.append(currentfreq)
            
            if len(currentdata) != 0:
                newdata.append(mean(currentdata))
            else:
                newdata.append(0)
            
        else:
            if magnitude[i] > low_threshold_samples:
                currentdata.append(magnitude[i])
    return newfreq, newdata

# Plot for Time of Robot Moving
plt.figure(0)
plt.plot(time, data)
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.title("Time Domain")

# Plot for FFT of Robot Moving
plt.figure(1)
if binFreqFlag == True:    
    freq1, data1 = binFreq(fftx[lowindex:highindex],abs(ffty[lowindex:highindex]),low_hz)
else:
    freq1 = fftx[lowindex:highindex]
    data1 = abs(ffty[lowindex:highindex])
   
if semilogFlag == True:
    plt.semilogx(freq1, data1)
else:
    plt.plot(freq1, data1)
    
plt.xlabel("Freq [Hz]")
plt.ylabel("Amplitude")
plt.title("Frequency Domain -- Raw Data")

# Plot for FFT of Robot Moving
plt.figure(2)

if binFreqFlag == True:    
    freq2, data2 = binFreq(noisex[lowindex:highindex],abs(noisey[lowindex:highindex]),low_hz)
else:
    freq2 = noisex[lowindex:highindex]
    data2 = abs(noisey[lowindex:highindex])
   
if semilogFlag == True:
    plt.semilogx(freq2, data2)
else:
    plt.plot(freq2, data2)

plt.xlabel("Freq [Hz]")
plt.ylabel("Amplitude")
plt.title("Frequency Domain -- Noise Data")

# Plot for FFT of Robot Moving
# Assuming Audio Clips are Same Length
plt.figure(3)
   
data3 = np.abs(np.abs(data1) - np.abs(data2))

if semilogFlag == True:
    plt.semilogx(freq1, data3)
else:
    plt.plot(freq1, data3)

plt.xlabel("Freq [Hz]")
plt.ylabel("Amplitude")
plt.title("Frequency Domain -- Filtered Out Noise")


plt.show()




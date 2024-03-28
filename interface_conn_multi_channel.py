# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 11:40:56 2024

@author: bmkea
"""

import sys
import pyaudio
import wave
import numpy as np

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 8
RATE = 48000
RECORD_SECONDS = 21

pyaudio_instance = pyaudio.PyAudio()


# find the index of respeaker usb device
def find_device_index():
    found = -1
    for i in range(pyaudio_instance.get_device_count()):
        dev = pyaudio_instance.get_device_info_by_index(i)
        name = dev['name'].encode('utf-8')
        print(i, name, dev['maxInputChannels'], dev['maxOutputChannels'])
        if name.find(b'IN 01-08 (BEHRINGER UMC 1820)') >= 0 and dev['maxInputChannels'] > 0:
            found = i
            break

    return found


device_index = find_device_index()
if device_index < 0:
    print('Channels 1-8 on Behringer not found')
    sys.exit(1)


stream = pyaudio_instance.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=CHUNK)

print("* recording for", RECORD_SECONDS, "seconds..." )

chan_1_frames = []
chan_2_frames = []
chan_3_frames = []
chan_4_frames = []
chan_5_frames = []
chan_6_frames = []
chan_7_frames = []
chan_8_frames = []

for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    #print(i)
    data = stream.read(CHUNK)
    #print(data)

    # convert string to numpy array
    data_array = np.fromstring(data, dtype='int16')

    # deinterleave, select 1 channel
    channel0 = data_array[0::CHANNELS]
    channel1 = data_array[1::CHANNELS]
    channel2 = data_array[2::CHANNELS]
    channel3 = data_array[3::CHANNELS]
    channel4 = data_array[4::CHANNELS]
    channel5 = data_array[5::CHANNELS]
    channel6 = data_array[6::CHANNELS]
    channel7 = data_array[7::CHANNELS]

    # convert numpy array to string 
    data = channel0.tostring()
    chan_1_frames.append(data)
    
    data = channel1.tostring()
    chan_2_frames.append(data)
    
    data = channel2.tostring()
    chan_3_frames.append(data)
    
    data = channel3.tostring()
    chan_4_frames.append(data)
    
    data = channel4.tostring()
    chan_5_frames.append(data)
    
    data = channel5.tostring()
    chan_6_frames.append(data)
    
    data = channel6.tostring()
    chan_7_frames.append(data)
    
    data = channel7.tostring()
    chan_8_frames.append(data)
    
    

print("* done recording")

stream.stop_stream()
stream.close()
pyaudio_instance.terminate()

wf = wave.open(r'C:\Users\bmkea\Documents\Denso_Test_cell\Python Scripts\Audio_Prediction\Audio_Files\audio_interface\7_robot2_base_output_ch1.wav', 'wb')
wf.setnchannels(1)
wf.setsampwidth(pyaudio_instance.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(chan_1_frames))
wf.close()

wf = wave.open(r'C:\Users\bmkea\Documents\Denso_Test_cell\Python Scripts\Audio_Prediction\Audio_Files\audio_interface\7_robot2_j1_output_ch2.wav', 'wb')
wf.setnchannels(1)
wf.setsampwidth(pyaudio_instance.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(chan_2_frames))
wf.close()

wf = wave.open(r'C:\Users\bmkea\Documents\Denso_Test_cell\Python Scripts\Audio_Prediction\Audio_Files\audio_interface\7_robot4_j1_output_ch3.wav', 'wb')
wf.setnchannels(1)
wf.setsampwidth(pyaudio_instance.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(chan_3_frames))
wf.close()

wf = wave.open(r'C:\Users\bmkea\Documents\Denso_Test_cell\Python Scripts\Audio_Prediction\Audio_Files\audio_interface\7_robot4_base_output_ch4.wav', 'wb')
wf.setnchannels(1)
wf.setsampwidth(pyaudio_instance.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(chan_4_frames))
wf.close()

wf = wave.open(r'C:\Users\bmkea\Documents\Denso_Test_cell\Python Scripts\Audio_Prediction\Audio_Files\audio_interface\7_robot3_j1_output_ch5.wav', 'wb')
wf.setnchannels(1)
wf.setsampwidth(pyaudio_instance.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(chan_5_frames))
wf.close()

wf = wave.open(r'C:\Users\bmkea\Documents\Denso_Test_cell\Python Scripts\Audio_Prediction\Audio_Files\audio_interface\7_robot3_base_output_ch6.wav', 'wb')
wf.setnchannels(1)
wf.setsampwidth(pyaudio_instance.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(chan_6_frames))
wf.close()

wf = wave.open(r'C:\Users\bmkea\Documents\Denso_Test_cell\Python Scripts\Audio_Prediction\Audio_Files\audio_interface\7_robot1_j1_output_ch7.wav', 'wb')
wf.setnchannels(1)
wf.setsampwidth(pyaudio_instance.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(chan_7_frames))
wf.close()

wf = wave.open(r'C:\Users\bmkea\Documents\Denso_Test_cell\Python Scripts\Audio_Prediction\Audio_Files\audio_interface\7_robot1_base_output_ch8.wav', 'wb')
wf.setnchannels(1)
wf.setsampwidth(pyaudio_instance.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(chan_8_frames))
wf.close()

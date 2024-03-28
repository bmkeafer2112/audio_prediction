# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 09:18:12 2024

@author: bmkea
"""

#Used to read vibration sensors
import sys
import pyaudio
import wave
import numpy as np
import pandas as pd

#Used to convert .wav to spectogram
from Denso_Audio import AudioUtil

class vibration():
    """
    This class is to read peizo-electric sensors that are connected through audio interface
    and utilize an FFT transformation to create a spectogram
    """
    def __init__(self):
        print("Building Spectogram")
        
    def record_vibration(self, chunk, channels, rate, record_time):
        """
        This method is to record vibration data from sensors that are connected through an audio interface
        and save this recording in a .wav format
        """
        
        #Set parameters for recording
        CHUNK = chunk
        CHANNELS = channels
        RATE = rate
        RECORD_SECONDS = record_time
        FORMAT = pyaudio.paInt16
        
        #Begin Recording Instance
        pyaudio_instance = pyaudio.PyAudio()
        
        # find the index of the behringer audio interface
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

        #Check if interface exists, it it exists begin recording
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

        print("* Recording for", RECORD_SECONDS, "seconds..." )
        
        #Create list to hold each chunk (frame) that is recorded for each channel
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

            # deinterleave each channel
            channel0 = data_array[0::CHANNELS]
            channel1 = data_array[1::CHANNELS]
            channel2 = data_array[2::CHANNELS]
            channel3 = data_array[3::CHANNELS]
            channel4 = data_array[4::CHANNELS]
            channel5 = data_array[5::CHANNELS]
            channel6 = data_array[6::CHANNELS]
            channel7 = data_array[7::CHANNELS]

            # convert numpy array to string  for each channel
            
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
            
        print("* Recording Complete")

        stream.stop_stream()
        stream.close()
        pyaudio_instance.terminate()
        
        #Read file that maps sensors to locations
        sensor_map = pd.read_csv(r'C:\Users\bmkea\Documents\Denso_Test_cell\Python Scripts\Audio_Prediction\Audio_Files\audio_interface\sensor_mapping.csv')
        file_list = []
        audio_data_dict = {'0': [],'1': [], '2': [], '3': [], '4': [], '5': [], '6': [], '7': []}
        audio_data_dict['0'].append(chan_1_frames)
        audio_data_dict['1'].append(chan_2_frames)
        audio_data_dict['2'].append(chan_3_frames)
        audio_data_dict['3'].append(chan_4_frames)
        audio_data_dict['4'].append(chan_5_frames)
        audio_data_dict['5'].append(chan_6_frames)
        audio_data_dict['6'].append(chan_7_frames)
        audio_data_dict['7'].append(chan_8_frames)
        
        
        for ind in sensor_map.index:
            frame = audio_data_dict.get(str(ind))
            print(sensor_map['input'][ind], sensor_map['robot'][ind], sensor_map['position'][ind])
            audio_input = str(sensor_map['input'][ind])
            robot = sensor_map['robot'][ind]
            sensor_position = sensor_map['position'][ind]
            file_list.append(r'C:\Users\bmkea\Documents\Denso_Test_cell\Python Scripts\Audio_Prediction\Audio_Files\audio_interface\21_robot%s_position%s_ch%s.wav' % (robot, sensor_position, audio_input))
            wf = wave.open(r'C:\Users\bmkea\Documents\Denso_Test_cell\Python Scripts\Audio_Prediction\Audio_Files\audio_interface\21_robot%s_position%s_ch%s.wav' % (robot, sensor_position, audio_input), 'wb')
            wf.setnchannels(1)
            wf.setsampwidth(pyaudio_instance.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frame[0]))
            wf.close()
            
        return file_list, chan_7_frames, frame
            
    def build_melspectogram(self, wave_files):
        """
        This method is to take a list of wave files transform them using FFT,
        scale them using melscale and decibel scale, and store the melspectogram
        """
        audio = AudioUtil()
        
        #Iterate over list of wave files that will be converted to a melspectogram
        for w in wave_files:
            
            file= audio.open(w)

            #file= audio.open(r'C:\Users\bmkea\Documents\Denso_Test_cell\Python Scripts\Audio_Prediction\Audio_Files\audio_interface\1_robot1_base_output_ch8.wav')
    
            #Ensure file is stereo
            new_channel = audio.rechannel(file, 1)
            
            #Ensure file is same sample rate (will upsample or downsample accordingly)
            resample = audio.resample(new_channel, 48000)
            
            #Make sure each file is 21 seconds
            new_length = audio.pad_trunc(resample, 21000)
            
            #Get Waveform and Spectogram data
            sig01, sr01 = new_length
            
            #Graph waveform and melspectogram
            audio.plot_waveform(sig01, sr01)
            audio.plot_mel_spectogram(sig01, sr01)




        
#!/usr/bin/env python
# coding: utf-8

# In[5]:


import matplotlib.pyplot as plt
import librosa
import librosa.display
import os
from os.path import isfile
import shutil
import gc
import warnings 
import numpy as np
import random
import string
warnings.filterwarnings("ignore")


# In[8]:


def randomString(stringLength=40):
    """Generate a random string of fixed length """
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(stringLength))



def get_length(path):
    y, sr  = librosa.load(path)
    length = librosa.get_duration(y=y, sr=sr)
    gc.collect()
    return length


# In[9]:


def save_wav(path_png, classname, data_path, num):
    length = get_length(data_path)
    
    for offset in range(0,int(length),num):
        fig = plt.figure(figsize=(5.04, 2.16))
        ax = plt.gca()
        ax.axis('off')

        # load type         
        y, sr = librosa.load(data_path,offset=offset, duration=10)   
        X = librosa.stft(y)
        Xdb = librosa.amplitude_to_db(abs(X))
        librosa.display.specshow(Xdb, sr=sr, y_axis='hz', x_axis="off",ax=ax)

        fig.savefig ( path_png + classname +"_"+str(num)+"_"+str(offset) )
        plt.close(fig)
        gc.collect()

def save_wav_only_vocal(path_png, classname, data_path, num):
    length = get_length(data_path)
    
    for offset in range(0,int(length),num):
        fig = plt.figure(figsize=(5.04, 2.16))
        ax = plt.gca()
        ax.axis('off')

        # load type         
        y, sr = librosa.load(data_path,offset=offset, duration=10)
        S_full, phase = librosa.magphase(librosa.stft(y))
        librosa.display.specshow(librosa.amplitude_to_db(S_full, ref=np.max),
                         y_axis='hz', x_axis='off', sr=sr, ax=ax)
      
        fig.savefig ( path_png + classname +"_only_vocal_"+randomString() )
        plt.close(fig)
        gc.collect()

def save_wav_only_vocal_noise(path_png, classname, data_path, num, noise):
    length = get_length(data_path)
    
    for offset in range(0,int(length),num):
        fig = plt.figure(figsize=(5.04, 2.16))
        ax = plt.gca()
        ax.axis('off')

        # load type         
        y, sr = librosa.load(data_path,offset=offset, duration=10)
        y = add_noise(y, noise)
        S_full, phase = librosa.magphase(librosa.stft(y))
        librosa.display.specshow(librosa.amplitude_to_db(S_full, ref=np.max),
                         y_axis='hz', x_axis='off', sr=sr, ax=ax)
      
        fig.savefig ( path_png + classname +"_only_vocal_noise_"+randomString() )
        plt.close(fig)
        gc.collect()

def save_wav_only_vocal_pitch(path_png, classname, data_path, num):
    length = get_length(data_path)
    
    for offset in range(0,int(length),num):
        fig = plt.figure(figsize=(5.04, 2.16))
        ax = plt.gca()
        ax.axis('off')

        # load type         
        y, sr = librosa.load(data_path,offset=offset, duration=5)
        y = manipulate(y, sr)
        S_full, phase = librosa.magphase(librosa.stft(y))
        librosa.display.specshow(librosa.amplitude_to_db(S_full, ref=np.max),
                         y_axis='hz', x_axis='off', sr=sr, ax=ax)
      
        fig.savefig ( path_png + classname +"_only_vocal_pitch_"+randomString() )
        plt.close(fig)
        gc.collect()


# In[10]:

def add_noise(data, noise_factor):
    noise = np.random.randn(len(data))
    data_noise = data + noise_factor * noise
    return data_noise

def manipulate(data, sampling_rate):
    bins_per_octave = 12
    pitch_pm = 2
    pitch_change =  pitch_pm * 2*(np.random.uniform())
    return librosa.effects.pitch_shift(data, sampling_rate, pitch_change)



def get_music(path_png, music_name, data_path):
    class_names = os.listdir(data_path)
    for idx, classname in enumerate(class_names):
        save_wav_only_vocal(path_png+"/"+music_name +"/",classname,data_path+"/"+classname, 3)    
        save_wav_only_vocal_noise(path_png+"/"+music_name +"/",classname,data_path+"/"+classname, 3, 0.008)
        save_wav_only_vocal_pitch(path_png+"/"+music_name +"/",classname,data_path+"/"+classname, 3)
        gc.collect()
# In[11]:


path="/home/changgeonlee/Project/DeepMusic"
path_data=path+"/data_wav/"
path_png=path+"/data_png_train/"


# In[7]:


# shutil.rmtree(path_png)
os.mkdir(path_png)


class_names = os.listdir(path_data)
for idx, classname in enumerate(class_names):
    data_path = path_data+classname
    print(str(idx+1) + "생성 중 : " +classname)
    try:
        # 이미지 저장할 폴더 생성         
        os.mkdir(path_png+classname)
        get_music(path_png,classname,data_path)
        gc.collect()
    except FileExistsError:
        print ("이미 생성됨")    
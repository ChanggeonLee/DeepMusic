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

def randomString(stringLength=40):
    """Generate a random string of fixed length """
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(stringLength))

def get_length(path):
    y, sr  = librosa.load(path)
    length = librosa.get_duration(y=y, sr=sr)
    gc.collect()
    return length


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
                         y_axis='hz', x_axis='off', sr=11025, ax=ax)
      
        fig.savefig ( path_png + classname +"_only_vocal_"+randomString() )
        plt.close(fig)
        gc.collect()

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
        save_wav_only_vocal(path_png+"/"+music_name +"/",classname,data_path+"/"+classname, 5)   
        save_wav_only_vocal(path_png+"/"+music_name +"/",classname,data_path+"/"+classname, 10)
        gc.collect()

path="/home/changgeonlee/Project/DeepMusic"
path_data=path+"/train_wav/"
path_png=path+"/data_png/"

# shutil.rmtree(path_png)
os.mkdir(path_png)

class_names = os.listdir(path_data)
for idx, classname in enumerate(class_names):
    data_path = path_data+classname
    print(str(idx+1) + "생성 중 : " +classname)
    try:    
        os.mkdir(path_png+classname)
        get_music(path_png,classname,data_path)
        gc.collect()
    except FileExistsError:
        print ("이미 생성됨")    
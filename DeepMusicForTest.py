#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import print_function
import os
import numpy as np
import cv2
import os
import sys
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
import keras
sys.stderr = stderr
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras.models import model_from_json
from keras.models import Sequential
from keras import losses
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.optimizers import Adam


# In[2]:


def get_total(data_path):
    subdir = os.listdir(data_path)
    return len(subdir)


# In[3]:


def load_model(path="./model/"):
    json_file = open(path+"DeepMusic.json", "r")
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(path+"DeepMusic.h5")
    print("Loaded model from disk")

    adam = keras.optimizers.Adam(learning_rate=0.001)
    loaded_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=adam, metrics=['accuracy'])
    return loaded_model


# In[4]:


def load_data_test(data_path):
    x_data = []
    subdir = os.listdir(data_path)
    for idex, subdir in enumerate(subdir):
#         print(data_path+subdir)
        img = cv2.imread(data_path+subdir,0)
        x_data.append(img/255)

    x_data = np.array(x_data)
    height = 216
    width = 504
    
    x_data = x_data.reshape(x_data.shape[0], height, width, 1)
    x_data = x_data.astype('float32')/255
    
    return x_data


# In[5]:


def test_load_data():
    height = 216
    width = 504
    
    x_data = []
    y_data = []

    data_path = "./test_data_png"
    total = get_total(data_path)
    subdir = os.listdir(data_path)

    for idex, subdir in enumerate(subdir):
        label = [0 for i in range(total)]
        label[idex] = 1
        for item in os.listdir(data_path+"/"+subdir):
            img_path = data_path+"/"+subdir+"/"+item
            img = cv2.imread(img_path,0)
            x_data.append(img/255)
            y_data.append(label)

    x_data = np.array(x_data)
    y_data = np.array(y_data)

#     x_train, y_train = shuffle_XY_paths(x_data, y_data)
    y_train = y_data
    x_train = x_data.reshape(x_data.shape[0], height, width, 1)

    x_train = x_train.astype('float32')/255
    
    input_shape = (height, width,1)

    return x_train, y_train


# In[6]:


model = load_model()
x_test, y_test = test_load_data()
print(model.evaluate(x_test, y_test, verbose=0))


# In[7]:


label = os.listdir("./data_png")
x = load_data_test(data_path="./test_data_png/방에_모기가_있어_10cm/")

y_label = model.predict(x)
for y in y_label:
    print(label[np.argmax(y)])


# In[8]:


print(label)


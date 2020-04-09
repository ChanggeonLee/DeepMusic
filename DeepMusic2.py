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

from keras.models import Sequential
from keras import losses
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import BatchNormalization, Conv2D, Activation, Dense, GlobalAveragePooling2D, MaxPooling2D, ZeroPadding2D, Add
from keras.layers import Conv2D, MaxPooling2D, Activation
from keras import backend as K
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Concatenate
from keras.layers import Add
from keras.layers import Input
from keras.utils import plot_model
from keras.layers import Activation
import tensorflow as tf
from keras.applications.resnet50 import ResNet50, decode_predictions
# import os
from os.path import isfile
import shutil
import gc
import warnings 
# import sys
warnings.filterwarnings("ignore")

def build_model(input_shape):
    nb_filters = 32  # number of convolutional filters to use
    pool_size = (2, 2)  # size of pooling area for max pooling
    kernel_size =(2,2)  # convolution kernel size
    nb_layers = 4
    total = 10
    
    # make model     
    model = Sequential()
    
    # conv layer
    model.add(Conv2D(16, kernel_size, 
                     padding='valid', 
                     input_shape=input_shape, 
                     activation='relu'))
#    model.add(Conv2D(32, kernel_size, use_bias=False, activation='relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    keras.layers.Dropout(rate=0.2)
   

    model.add(Conv2D(32, kernel_size, use_bias=False, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
   # model.add(Conv2D(64, kernel_size,activation='relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    keras.layers.Dropout(rate=0.2)
    

    model.add(Conv2D(64, kernel_size, use_bias=False, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
   # model.add(Conv2D(64, kernel_size,activation='relu'))
    model.add(MaxPooling2D(pool_size=pool_size, padding='same'))
    keras.layers.Dropout(rate=0.2)
    
    

    model.add(Conv2D(128, kernel_size, use_bias=False, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
   # model.add(Conv2D(128, kernel_size,activation='relu'))
    model.add(MaxPooling2D(pool_size=pool_size,padding='same'))
    keras.layers.Dropout(rate=0.2)



    model.add(Conv2D(256, kernel_size, use_bias=False, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=pool_size))
    keras.layers.Dropout(rate=0.2)
    model.add(Flatten())

   # model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))
       
    adam = keras.optimizers.Adam(learning_rate=0.00005)
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=adam,
                  metrics=['accuracy'])

    return model


# In[20]:


def save_model(model, path="/home/jang/DeepMusic/DeepMusic/model/"):
    model_json = model.to_json()
    with open(path+"DeepMusic.json", "w") as json_file : 
        json_file.write(model_json)
    model.save_weights(path+"DeepMusic.h5")
    print("Saved model to disk")

def load_data():
    x_train = np.load("./data/x_train.npy")
    y_train = np.load("./data/y_train.npy")
    x_test = np.load("./data/x_test.npy")
    y_test = np.load("./data/y_test.npy")

    return x_train, y_train, x_test, y_test


x_train, y_train, x_test, y_test = load_data()
input_shape =  (216, 504,3)
# x_test, y_test = test_load_data()
print("input shape : ",input_shape)
print("x train : ", x_train.shape)
print("y train : ", y_train.shape)
print("x test : ", x_test.shape)
print("y test : ", y_test.shape)


# In[16]:


model = build_model(input_shape)
model.summary()


model.fit(x_train, y_train,
          batch_size=20,
          epochs=30,
          verbose=1,
          validation_data=(x_test, y_test),
          shuffle=True)

save_model(model)

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
from keras.layers import Conv2D, MaxPooling2D, Activation
from keras import backend as K
from keras.optimizers import Adam
# shutil.rmtree('/home/changgeonlee/Project/DeepMusic/test_data_png/.ipynb_checkpoints')
# shutil.rmtree('/home/changgeonlee/Project/DeepMusic/data_png/.ipynb_checkpoints')


# In[2]:


def get_total(data_path):
    subdir = os.listdir(data_path)
    return len(subdir)


# In[3]:


def shuffle_XY_paths(X,Y): 
    assert (X.shape[0] == Y.shape[0] )
    idx = np.array(range(Y.shape[0]))
    np.random.shuffle(idx)
    newX = np.copy(X)
    newY = np.copy(Y)
    
    for i in range(len(idx)):
        newX[i] = X[idx[i],:,:]
        newY[i] = Y[idx[i],:]

    return newX, newY


# In[4]:


def load_data():
    height = 216
    width = 504
    
    x_data = []
    y_data = []

    data_path = "./data_png"
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

    x_train, y_train = shuffle_XY_paths(x_data, y_data)

    x_train = x_train.reshape(x_train.shape[0], height, width, 1)

    x_train = x_train.astype('float32')/255
    
    input_shape = (height, width,1)

    return x_train, y_train,input_shape


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

    x_train, y_train = shuffle_XY_paths(x_data, y_data)

    x_train = x_train.reshape(x_train.shape[0], height, width, 1)

    x_train = x_train.astype('float32')/255
    
    input_shape = (height, width,1)

    return x_train, y_train


# In[6]:


def build_model(input_shape):
    nb_filters = 32  # number of convolutional filters to use
    pool_size = (2, 2)  # size of pooling area for max pooling
    kernel_size = (3, 3)  # convolution kernel size
    nb_layers = 4
    total = 10
    
    # make model     
    model = Sequential()
    
    # conv layer
    model.add(Conv2D(32, kernel_size, 
                     padding='valid', 
                     input_shape=input_shape, 
                     activation='relu'))
    model.add(Conv2D(32, kernel_size, activation='relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    keras.layers.Dropout(rate=0.25)
    
    model.add(Conv2D(32, kernel_size, use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=pool_size))
    keras.layers.Dropout(rate=0.25)
    
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    keras.layers.Dropout(rate=0.5)
    model.add(Dense(10, activation='softmax'))
    
    adam = keras.optimizers.Adam(learning_rate=0.0005)
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=adam,
                  metrics=['accuracy'])

    return model


# In[7]:


def save_model(model, path="./model/"):
    model_json = model.to_json()
    with open(path+"DeepMusic.json", "w") as json_file : 
        json_file.write(model_json)
    model.save_weights(path+"DeepMusic.h5")
    print("Saved model to disk")


# In[8]:


x_train, y_train, input_shape = load_data()
x_test, y_test = test_load_data()
print("input shape : ",input_shape)
print("x train : ", x_train.shape)
print("y train : ", y_train.shape)
print("x test : ", x_test.shape)
print("y test : ", y_test.shape)


# In[ ]:


model = build_model(input_shape)
model.summary()


# In[ ]:


model.fit(x_train, y_train,
          batch_size=40,
          epochs=40,
          verbose=1,
          validation_data=(x_test, y_test),
          shuffle=True)


# In[ ]:


save_model(model)


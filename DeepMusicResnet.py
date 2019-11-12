#!/usr/bin/env python
# coding: utf-8

# In[19]:


pip install pydot


# In[1]:


from __future__ import print_function
import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import shutil
import tensorflow as tf
import keras
from keras.models import Sequential
from keras import losses
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Activation
from keras import backend as K
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Concatenate
from keras.layers import Add
from keras.layers import Input
from keras.utils import plot_model
from keras.layers import Activation


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

#     shutil.rmtree('./data_png/.ipynb_checkpoints', ignore_errors=True)
    for idex, subdir in enumerate(subdir):
        label = [0 for i in range(total)]
        label[idex] = 1
#         print(subdir)
#         print(label)
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

#     shutil.rmtree('./.ipynb_checkpoints', ignore_errors=True)
    for idex, subdir in enumerate(subdir):
        label = [0 for i in range(total)]
        label[idex] = 1
#         print(subdir)
#         print(label)
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


# In[18]:


def build_model(input_shape):
    nb_filters = 32  # number of convolutional filters to use
    pool_size = (2, 2)  # size of pooling area for max pooling
    kernel_size = (3, 3)  # convolution kernel size
    nb_layers = 4
    total = 10
    
    #resnet
    visible = Input(shape=input_shape)
    conv1 = Conv2D(16, kernel_size=3, activation='relu',padding='same')(visible)
#     batch = BatchNormalization(momentum=0.99)(conv1)
#     active = Activation(activation='relu')(batch)
    conv2 = Conv2D(64, kernel_size=3, activation='relu',padding='same')(conv1)
    conv3 = Conv2D(128, kernel_size=3, activation='relu',padding='same')(conv2)
    add1 = Add()([visible, conv3])

    conv4 = Conv2D(128, kernel_size=3,activation='relu',padding='same')(add1)
#     batch2 = BatchNormalization(momentum=0.99)(conv4)
#     active2 = Activation(activation='relu')(batch2)
    conv5 = Conv2D(128, kernel_size=3, activation='relu',padding='same')(conv4)
    conv6 = Conv2D(128, kernel_size=3, activation='relu',padding='same')(conv5)
    add2 = Add()([add1, conv6])

    conv7 = Conv2D(128, kernel_size=3,activation='relu',padding='same')(add2)
#     batch2 = BatchNormalization(momentum=0.99)(conv4)
#     active2 = Activation(activation='relu')(batch2)
    conv8 = Conv2D(128, kernel_size=3, activation='relu',padding='same')(conv7)
    conv9 = Conv2D(128, kernel_size=3, activation='relu',padding='same')(conv8)
    add3 = Add()([add2, conv9])
    
    dense1 = Dense(64, activation='relu')(add3)
    dense2 = Dense(32, activation='relu')(dense1)
   
    flat = Flatten()(dense2)
    
    output = Dense(11, activation='softmax')(flat)
    model = Model(inputs=[visible], outputs=[output])



#     model = Sequential()


    adam = keras.optimizers.Adam(lr=0.005)

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


# In[19]:


model = build_model(input_shape)
model.summary()
plot_model(model, to_file='inception.png')


# In[ ]:


model.fit(x_train, y_train,
          batch_size=5,
          epochs=50,
          verbose=1,
          validation_data=(x_test, y_test))


# In[45]:


save_model(model)


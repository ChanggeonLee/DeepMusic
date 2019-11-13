import os
import numpy as np
import cv2
import os
import sys
import gc


def get_total(data_path):
    subdir = os.listdir(data_path)
    return len(subdir)

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


def load_data():
    height = 216
    width = 504
    
    x_data = []
    y_data = []

    data_path = "/home/changgeonlee/Project/DeepMusic/data_png"
    total = get_total(data_path)
    subdir = os.listdir(data_path)

    for idex, subdir in enumerate(subdir):
        label = [0 for i in range(total)]
        label[idex] = 1

        for item in os.listdir(data_path+"/"+subdir):
            img_path = data_path+"/"+subdir+"/"+item
            img = cv2.imread(img_path,0)
            gc.collect()
            x_data.append(img/255)
            y_data.append(label)

    x_data = np.array(x_data)
    y_data = np.array(y_data)

    x_train, y_train = shuffle_XY_paths(x_data, y_data)

    x_train = x_train.reshape(x_train.shape[0], height, width, 1)

    x_train = x_train.astype('float32')/255
    
    input_shape = (height, width,1)

    return x_train, y_train

def test_load_data():
    height = 216
    width = 504
    
    x_data = []
    y_data = []

    data_path = "/home/changgeonlee/Project/DeepMusic/test_data_png"
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


# In[11]:

def save_data(x_train, y_train, x_test, y_test):
    np.save("../data/x_train.npy",x_train)
    np.save("../data/y_train.npy",y_train)
    np.save("../data/x_test.npy",x_test)
    np.save("../data/y_test.npy",y_test)


x_train, y_train = load_data()
x_test, y_test = test_load_data()
save_data(x_train, y_train, x_test, y_test)

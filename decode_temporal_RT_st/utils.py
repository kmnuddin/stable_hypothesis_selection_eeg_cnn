import matplotlib.pyplot as plt

from scipy.ndimage.interpolation import zoom
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow.keras.backend as K
from tqdm import tqdm
import numpy as np
import os
import re
from multiprocessing import Process
from scipy import ndimage
import cv2
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from sklearn.preprocessing import MinMaxScaler

import psutil

import gradcamutils

def is_memory_overflow():
    if float(psutil.virtual_memory()._asdict()['percent'] >= 90.0):
        return True
    else:
        return False

def predict(model, X):
    return np.argmax(model.predict(X), axis=1)

def get_data(path, X_i, labels):
    slow = np.zeros((5972, 64, 500))
    med = np.zeros((6302, 64, 500))
    fast = np.zeros((6227, 64, 500))
    s_i = 0
    m_i = 0
    f_i = 0
    for i, id in tqdm(enumerate(X_i)):
        s_path = os.path.join(path, "{}.npy".format(id))
        data = np.load(s_path)

        if labels[id - 1] == 0:
            slow[s_i] = data
            # if s_i == 0:
            #     slow[s_i] = data
            # else:
            #     slow = np.append(slow, data, axis=0)
            #
            s_i += 1
            continue

        if labels[id - 1] == 1:
            med[m_i] = data
            # if m_i == 0:
            #     med[m_i] = data
            # else:
            #     med = np.append(med, data, axis=0)
            #
            m_i += 1
            continue

        if labels[id -1] == 2:
            fast[f_i] = data
            # if f_i == 0:
            #     fast[f_i] = data
            # else:
            #     fast = np.append(fast, data, axis=0)
            #
            f_i += 1
            continue

    print("slow: {}, med: {}, fast: {}".format(s_i, m_i, f_i))

    slow = np.swapaxes(slow, 1, 2)
    med = np.swapaxes(med, 1, 2)
    fast = np.swapaxes(fast, 1, 2)

    return slow, med, fast

def get_saliency_maps(model, X, Y, t=500, ch=64, batch_size=1000, layer='conv2d_13'):
    gradcam = np.empty((X.shape[0], t, ch))
    N = len(X)
    for i in tqdm(range((N + batch_size - 1) // batch_size)):
        status = is_memory_overflow()
        if status:
            raise MemoryOverflow
        start = i * batch_size
        end = min((i+1) * batch_size, N)
        gradcam[start:end] = gradcamutils.grad_cam_batch(model, X[start:end], Y[start:end], layer, t, ch)


    return gradcam

def find_rel_layer(model):
    rel_layer = None
    for layer in model.layers[::-1]:

        if 'conv1d' in layer.name:
            rel_layer = layer.name
            break

    return rel_layer

def normalize(data):
    data = data.astype(np.float64)
    max_val = data.max()
    min_val = data.min()
    data = (data - min_val)/(max_val - min_val)
    return data


class MemoryOverflow(Exception):

    def __init__(self, message=" Please restart the process, 85 percent of the memory is already been used. There is chance of memory overflow."):
        self.message = message

        super().__init__(self.message)

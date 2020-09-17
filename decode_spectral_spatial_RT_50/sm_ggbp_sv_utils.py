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

def load_images(dir_path, H=128, W=128):
    files = os.listdir(dir_path)
    N = len(files)
    X = np.empty((N, H, W, 3))
    for i, file in tqdm(enumerate(files)):
        x = cv2.imread(dir_path + file)
        x = cv2.resize(x, (H,W))
        x = x/np.max(x)
        X[i] = x

    return X

def scale_img(img, qmin=0, qmax=255):
    max_val = img.max()
    min_val = img.min()
    scaled_img = ((img - min_val)/(max_val - min_val)) * (qmax - qmin)

def predict(model, X):
    Y = model.predict(X)
    return np.argamx(Y, axis=1)

def get_saliency_maps(model, X, Y, H=128, W=128, batch_size=256, layer='conv2d_13'):
    gradcam = np.empty((X.shape[0], H, W))
    N = len(X)
    for i in tqdm(range((N + batch_size - 1) // batch_size)):
        status = is_memory_overflow()
        if status:
            raise MemoryOverflow
        start = i * batch_size
        end = min((i+1) * batch_size, N)
        gradcam[start:end] = gradcamutils.grad_cam_batch(model, X[start:end], Y[start:end], layer, H, W)


    return gradcam


def get_guided_backprop_vals(guided_model, X, layer='conv2d_13'):
    gbp = np.empty((X.shape))
    batch_size = 160
    N = len(X)
    for i in tqdm(range((N + batch_size - 1) // batch_size)):
        status = is_memory_overflow()
        if status:
            raise MemoryOverflow
        start = i * batch_size
        end = min((i+1) * batch_size, N)
        gbp[start:end] = gradcamutils.guided_backprop(guided_model, X[start:end], layer)

    return gbp

def get_saliency_scores_from_ggbp(ggbp, pos, kernel):
    i, j = kernel
    sc = []
    for p in pos:
        x = p[0]
        y = p[1]

        alpha_w = ggbp[x-i//2:x+i//2, y-j//2:y+j//2, 0]
        beta_w = ggbp[x-i//2:x+i//2, y-j//2:y+j//2, 1]
        gamma_w = ggbp[x-i//2:x+i//2, y-j//2:y+j//2, 2]

        alpha = np.median(alpha_w)
        beta = np.median(beta_w)
        gamma = np.median(gamma_w)

        sc.extend([alpha, beta, gamma])

    return sc

def find_rel_layer(model):
    rel_layer = None
    for layer in model.layers[::-1]:

        if 'add' in layer.name or 'conv2d' in layer.name:
            rel_layer = layer.name
            break

    return rel_layer


class MemoryOverflow(Exception):

    def __init__(self, message=" Please restart the process, 85 percent of the memory is already been used. There is chance of memory overflow."):
        self.message = message

        super().__init__(self.message)

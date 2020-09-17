from scipy.ndimage.interpolation import zoom
import numpy as np

from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import load_img, img_to_array

import tensorflow as tf
import tensorflow.compat.v1 as tf1
from tensorflow.python.framework import ops
from tensorflow.keras.models import load_model

tf.compat.v1.disable_eager_execution()

import matplotlib.pyplot as plt
import cv2

def normalize(x):
        """Utility function to normalize a tensor by its L2 norm"""
        return (x + 1e-10) / (K.sqrt(K.mean(K.square(x))) + 1e-10)

def deprocess_image(x):
    """Same normalization as in:
    https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py
    """
    # normalize tensor: center on 0., ensure std is 0.25
    x = x.copy()
    x -= x.mean()
    x /= (x.std() + K.epsilon())
    x *= 0.25

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_data_format() == 'channels_first':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def grad_cam_batch(input_model, images, classes, layer_name, t=500, ch=64):
    """GradCAM method for visualizing input saliency.
    Same as grad_cam but processes multiple images in one run."""
    loss = tf.gather_nd(input_model.output, np.dstack([range(images.shape[0]), classes])[0])
    layer_output = input_model.get_layer(layer_name).output
    grads = K.gradients(loss, layer_output)[0]
    gradient_fn = K.function([input_model.input, K.learning_phase()], [layer_output, grads])

    conv_output, grads_val = gradient_fn([images, 0])

    weights = np.mean(grads_val, axis=(1, 2))


    cams = np.einsum('ijk,i->ij', conv_output, weights)
    # Process CAMs
    new_cams = np.empty((images.shape[0], t, ch))
    for i in range(new_cams.shape[0]):
        cam_i = cams[i] - cams[i].mean()
        cam_i = (cam_i + 1e-10) / (np.linalg.norm(cam_i, 2) + 1e-10)
        new_cams[i] = cv2.resize(cam_i, (ch, t), cv2.INTER_CUBIC)
        new_cams[i] = np.maximum(new_cams[i], 0)
        new_cams[i] = new_cams[i] / new_cams[i].max()

    return new_cams

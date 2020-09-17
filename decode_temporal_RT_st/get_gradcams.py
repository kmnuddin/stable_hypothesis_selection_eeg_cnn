from utils import get_data, predict, find_rel_layer, get_saliency_maps
import pandas as pd
import numpy as np
import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras.models import load_model
from tqdm import tqdm
from pympler import muppy, summary
import objgraph

import os


def saliency_maps(model_dirs, save_dir, X, rt_class):
    for i,dir in tqdm(enumerate(model_dirs)):
        K.clear_session()
        save_path = os.path.join(save_dir, rt_class, "TM-{}.npy".format(i+1))

        if os.path.exists(save_path):
            continue

        model = load_model(dir)
        layer = find_rel_layer(model)
        Y = predict(model, X)

        sm = get_saliency_maps(model, X, Y, t=500, ch=64, layer=layer)
        np.save(save_path, sm)
        tf.compat.v1.reset_default_graph()

X_indexes = pd.read_csv("data/X_test_10.csv", header=None).values.flatten()
Y_indexes = np.load("data/Y_RT_10.npy")

data_dir = "data/RT_10_data"

slow, med, fast = get_data(data_dir, X_indexes, Y_indexes)

data = [slow, med, fast]

model_dirs = [os.path.join("top_10_tm", "TM-{}.h5".format(i+1)) for i in range(10)]

rt_classes = ['slow', 'med', 'fast']

save_dir = "saliency_maps"



try:

    for i, X in enumerate(data):

        saliency_maps(model_dirs, save_dir, X, rt_classes[i])

except Exception as ex:
    biggest_vars = muppy.sort(muppy.get_objects())[-3:]
    objgraph.show_backrefs(biggest_vars, filename='backref.png')

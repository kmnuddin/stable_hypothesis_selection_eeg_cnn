from tensorflow.keras.models import load_model
from sm_ggbp_sv_utils import load_images, get_saliency_maps, predict, find_rel_layer
import os
from tqdm import tqdm
import numpy as np
import argparse
import tensorflow.keras.backend as K
from pympler import muppy, summary
import pandas as pd
import objgraph
import tensorflow as tf

def saliency_maps(model_dirs, save_dir, X):
    for i,dir in tqdm(enumerate(model_dirs)):
        K.clear_session()
        save_path = os.path.join(save_dir, "SPSM-{}.npy".format(i+1))

        if os.path.exists(save_path):
            continue

        model = load_model(dir)
        layer = find_rel_layer(model)

        Y = np.argmax(model.predict(X), axis=1)

        sm = get_saliency_maps(model, X, Y, layer=layer)
        np.save(save_path, sm)
        tf.compat.v1.reset_default_graph()





slow_path = 'data/topomaps_RT_50_w_sub/test/combined/0/'
med_path = 'data/topomaps_RT_50_w_sub/test/combined/3/'
fast_path = 'data/topomaps_RT_50_w_sub/test/combined/1/'

argparser = argparse.ArgumentParser()
argparser.add_argument('--RTClass', type=str)

args = argparser.parse_args()
rt_class = args.RTClass

path = None

if rt_class == 'slow':
    path = slow_path

elif rt_class == 'med':
    path = med_path

elif rt_class == 'fast':
    path = fast_path

model_dirs = [os.path.join("top_10_spsm", "SPSM-{}.h5".format(i+1)) for i in range(10)]

save_dir = os.path.join("saliency_maps", rt_class)

X = load_images(path)

try:

    saliency_maps(model_dirs, save_dir, X)

except Exception as ex:
    print(ex.message)
    biggest_vars = muppy.sort(muppy.get_objects())[-3:]
    objgraph.show_backrefs(biggest_vars, filename='backref.png')

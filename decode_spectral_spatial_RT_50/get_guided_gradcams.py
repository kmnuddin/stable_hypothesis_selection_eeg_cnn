from tensorflow.keras.models import load_model
from sm_ggbp_sv_utils import load_images, get_guided_backprop_vals, predict, find_rel_layer
import os
from tqdm import tqdm
import numpy as np
import argparse
import tensorflow.keras.backend as K
from pympler import muppy, summary
import pandas as pd
import objgraph
import tensorflow as tf
from tensorflow import Graph

import gradcamutils

def guided_gradcams(model_dirs, save_dir, X, sm_load_dir):
    indices = np.arange(0, len(X), 1000)
    indices = np.append(indices, len(X))
    for i,dir in tqdm(enumerate(model_dirs)):

        save_path = os.path.join(save_dir, "SPSM-{}.npy".format(i+1))

        if os.path.exists(save_path):
            continue
        gbp = np.empty(X.shape)

        for i in range(len(indices) - 1):
            K.clear_session()

            start = indices[i]
            end = indices[i+1]

            model = load_model(dir)

            guided_model = gradcamutils.build_guided_model(model)

            layer = find_rel_layer(model)

            gbp[start:end] = get_guided_backprop_vals(guided_model, X[start:end], layer=layer)

            tf.compat.v1.reset_default_graph()

        sm_load_path = os.path.join(sm_load_dir, "SPSM-{}.npy".format(i+1))

        sm = np.load(sm_load_path)



        ggbp = gbp * sm[..., np.newaxis]

        ggbp = np.where(ggbp < 0, 0, ggbp)

        np.save(save_path, ggbp)







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

save_dir = os.path.join("guided_gradcams", rt_class)

sm_load_dir = os.path.join("saliency_maps", rt_class)

X = load_images(path)

try:

    guided_gradcams(model_dirs, save_dir, X, sm_load_dir)

except Exception as ex:
    print(ex.message)

    all_objects = muppy.get_objects()
    sum1 = summary.summarize(all_objects)
    summary.print_(sum1)
    biggest_vars = muppy.sort(all_objects)[-3:]
    objgraph.show_backrefs(biggest_vars, filename='backref.png')

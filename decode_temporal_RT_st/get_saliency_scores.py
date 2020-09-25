import numpy as np
import argparse
from tqdm import tqdm
import os
from utils import normalize
from multiprocessing import Process

def saliency_scores(sm_dir, save_dir, rt_class, window, pre_st_ts):
    sm_paths = [os.path.join(sm_dir, rt_class, "TM-{}.npy".format(i+1)) for i in range(10)]
    for i,sm_path in enumerate(sm_paths):
        save_path = os.path.join(save_dir, rt_class, "TM-{}.npy".format(i+1))
        if os.path.exists(save_path):
            continue

        s_maps = normalize(np.load(sm_path))
        s_maps_t = np.mean(s_maps, axis=2)

        length = s_maps.shape[0]
        time_steps = s_maps.shape[1]

        ts_indexes = list(range(pre_st_ts, time_steps, window))
        ts_indexes.append(time_steps)
        ts_indexes.insert(0, 0)

        no_avg_ts = len(ts_indexes)
        sc = np.empty((length, no_avg_ts-1))

        for i in range(0, no_avg_ts):
            if i + 1 == no_avg_ts:
                break
            sc[:, i] = np.median(s_maps_t[:, ts_indexes[i]: ts_indexes[i+1]], axis=1)
        np.save(save_path, sc)

parser = argparse.ArgumentParser()
parser.add_argument('--window', type=int)
parser.add_argument('--pre_st_ts', type=int)

args = parser.parse_args()
window = args.window
pre_st_ts = args.pre_st_ts


sm_dir = 'saliency_maps'
rt_classes = ['slow', 'med', 'fast']

save_dir = 'saliency_scores/median'

processes = []

for rt_class in rt_classes:

    process = Process(target=saliency_scores, args=(sm_dir, save_dir, rt_class, window, pre_st_ts))

    processes.append(process)
    process.start()

for process in processes:
    process.join()

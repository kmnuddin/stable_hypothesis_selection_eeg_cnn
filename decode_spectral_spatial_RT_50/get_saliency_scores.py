from sm_ggbp_sv_utils import get_saliency_scores_from_ggbp
import numpy as np
import os
from multiprocessing import Process
import argparse

def saliency_scores(ggbp_dir, model_ids, rt_class, save_dir, pos, kernel):

    for id in model_ids:
        path = os.path.join(ggbp_dir, rt_class, "{}.npy".format(id))
        kernel_str = "{}x{}".format(kernel[0], kernel[1])
        save_path = os.path.join(save_dir, rt_class, "{}-{}.npy".format(id, kernel_str))

        if os.path.exists(save_path):
            continue
        ggbps = np.load(path)

        sc = np.empty((len(ggbps), 192))
        for i, ggbp in enumerate(ggbps):
            sc[i] = get_saliency_scores_from_ggbp(ggbp, pos, kernel)



        np.save(save_path, sc)


argparser = argparse.ArgumentParser()
argparser.add_argument('--kernel', nargs=2, type=int)

args = argparser.parse_args()
kernel = tuple(args.kernel)

print(kernel)

ggbp_dir = "guided_gradcams"

rt_classes = ['slow', 'med', 'fast']

model_ids = ["SPSM-{}".format(i+1) for i in range(10)]

pos = np.load("channel_pos_128x128.npy")

save_dir = "saliency_scores"


processes = []

for rt_class in rt_classes:

    process = Process(target=saliency_scores, args=(ggbp_dir, model_ids, rt_class, save_dir, pos, kernel))

    processes.append(process)
    process.start()

for process in processes:
    process.join()

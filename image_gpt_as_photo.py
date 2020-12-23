import h5py
import numpy as np
import os
import torch
from torch.utils import tensorboard

log_dir = '/home/dsi/eyalbetzalel/pytorch-generative-v6/image_test'
_summary_writer = tensorboard.SummaryWriter(log_dir, max_queue=100)


def load_h5_dataset(directory):
    print(" --------------------------------- ")
    print("Start loading Datasat from H5DF files...")
    data = []
    flagOneFile = 0
    for filename in os.listdir(directory):
        if flagOneFile:
            break
        if filename.endswith(".h5"):
            with h5py.File(filename, "r") as f:
                a_group_key = list(f.keys())[0]
                # Get the data
                temp = list(f[a_group_key])
                data.append(temp[1:])

                flagOneFile = 1
            continue
        else:
            continue
    data_flat = [item for sublist in data for item in sublist]
    data_flat = np.stack(data_flat, axis=0)
    precent_train_test_split = 0.7
    train = data_flat[:int(np.floor(precent_train_test_split * data_flat.shape[0])), :]
    test = data_flat[int(np.floor(precent_train_test_split * data_flat.shape[0])) + 1:, :]
    print(" --------------------------------- ")
    print("Finish loading Datasat from H5DF files...")

    return train, test

directory = "./"

train, test = load_h5_dataset(directory)

pathToCluster = r"/home/dsi/eyalbetzalel/image-gpt/downloads/kmeans_centers.npy"  # TODO : add path to cluster dir
global clusters
clusters = torch.from_numpy(np.load(pathToCluster)).float()

train = torch.reshape(train, [-1, 1, 32, 32])
sample = torch.reshape(torch.round(127.5 * (clusters[train.long()] + 1.0)), [train.shape[0], 3, 32, 32]).to('cuda')
_summary_writer.add_images("sample", sample)
import numpy as np
import numpy as np
import os
import torch
from torch.utils import tensorboard

pathToCluster = r"/home/dsi/eyalbetzalel/image-gpt/downloads/kmeans_centers.npy"  # TODO : add path to cluster dir
global clusters
clusters = torch.from_numpy(np.load(pathToCluster)).float()

def transform_cluster_to_image(data):
    log_dir = '/home/dsi/eyalbetzalel/pytorch-generative-v6/image_test'
    _summary_writer = tensorboard.SummaryWriter(log_dir, max_queue=100)
    data = torch.reshape(torch.from_numpy(train), [-1, 32, 32])
    # train = train[:,None,:,:]
    sample = torch.reshape(torch.round(127.5 * (clusters[data.long()] + 1.0)), [data.shape[0], 3, 32, 32]).to('cuda')
    _summary_writer.add_images("sample", sample)

def load_data(data_path):
    trX = np.load(f'{data_path}_trX.npy')
    trY = np.load(f'{data_path}_trY.npy')
    vaX = np.load(f'{data_path}_vaX.npy')
    vaY = np.load(f'{data_path}_vaY.npy')
    teX = np.load(f'{data_path}_teX.npy')
    teY = np.load(f'{data_path}_teY.npy')
    return (trX, trY), (vaX, vaY), (teX, teY)

data_path = './cifar10'
(trX, trY), (vaX, vaY), (teX, teY) = load_data(data_path)
transform_cluster_to_image(trX)
import ipdb; ipdb.set_trace()
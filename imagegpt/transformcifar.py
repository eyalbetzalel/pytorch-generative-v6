import numpy as np
import numpy as np
import os
import torch
from torch.utils import tensorboard
import torchvision

pathToCluster = r"/home/dsi/eyalbetzalel/image-gpt/downloads/kmeans_centers.npy"  # TODO : add path to cluster dir
global clusters
clusters = torch.from_numpy(np.load(pathToCluster)).float()

def transform_cluster_to_image(data):
    data = torch.reshape(torch.from_numpy(data), [-1, 32, 32])
    sample = torch.reshape(torch.round(127.5 * (clusters[data.long()] + 1.0)), [data.shape[0], 3, 32, 32]).to('cuda')
    return sample

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
sample = transform_cluster_to_image(trX)

log_dir = '/home/dsi/eyalbetzalel/pytorch-generative-v6/image_test/test.png'
# _summary_writer = tensorboard.SummaryWriter(log_dir = log_dir, max_queue=100)
# _summary_writer.add_images('sample',sample,0)
# _summary_writer.close()
torchvision.utils.save_image(torchvision.utils.make_grid(sample[1:48,:,:,:]), log_dir)




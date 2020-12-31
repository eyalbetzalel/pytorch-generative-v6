import numpy as np
import os
import torch
from torch.utils import tensorboard
import torchvision
import matplotlib.pyplot as plt


pathToCluster = r"/home/dsi/eyalbetzalel/image-gpt/downloads/kmeans_centers.npy"  # TODO : add path to cluster dir
global clusters
clusters = torch.from_numpy(np.load(pathToCluster)).float()

def transform_cluster_to_image(samples):

    data_tor = torch.reshape(torch.from_numpy(samples), [-1, 32, 32])
    clusters = torch.from_numpy(np.load(pathToCluster)).float()
    sample_new = torch.round(127.5 * (clusters[data_tor.long()] + 1.0))
    sample_new = sample_new.permute(0, 3, 1, 2)
    yos = torch.eq(samples,sample_new)

    return samples

def load_data(data_path):
    trX = np.load(f'{data_path}_trX.npy')
    trY = np.load(f'{data_path}_trY.npy')
    vaX = np.load(f'{data_path}_vaX.npy')
    vaY = np.load(f'{data_path}_vaY.npy')
    teX = np.load(f'{data_path}_teX.npy')
    teY = np.load(f'{data_path}_teY.npy')
    return (trX, trY), (vaX, vaY), (teX, teY)

def plot_images_grid(x: torch.tensor, export_img, title: str = '', nrow=8, padding=2, normalize=True, pad_value=0):
    """Plot 4D Tensor of images of shape (B x C x H x W) as a grid."""
    grid = torchvision.utils.make_grid(x, nrow=nrow, padding=padding, normalize=normalize, pad_value=pad_value)
    npgrid = grid.cpu().numpy()
    im = np.transpose(npgrid, (1, 2, 0))
    plt.imsave(export_img,im)


data_path = './cifar10'
(trX, trY), (vaX, vaY), (teX, teY) = load_data(data_path)
sample = transform_cluster_to_image(vaX)
log_dir = '/home/dsi/eyalbetzalel/pytorch-generative-v6/image_test/test1.png'
pytorch_tensor = sample[1:48,:,:,:]
plot_images_grid(pytorch_tensor, export_img=log_dir)






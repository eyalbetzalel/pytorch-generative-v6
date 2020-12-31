import numpy as np
import numpy as np
import os
import torch
from torch.utils import tensorboard
import torchvision
import matplotlib.pyplot as plt
import tensorflow as tf



pathToCluster = r"/home/dsi/eyalbetzalel/image-gpt/downloads/kmeans_centers.npy"  # TODO : add path to cluster dir
global clusters
clusters = torch.from_numpy(np.load(pathToCluster)).float()

def transform_cluster_to_image(samples):

    ############################### OLD ########################################################
    clusters =np.load(pathToCluster)
    samples = [torch.from_numpy(np.reshape(np.rint(127.5 * (clusters[s.astype(int).tolist()] + 1.0)), [3, 32, 32]).astype(np.float32)) for
               s in samples]
    samples = torch.stack(samples)


    ############################### OLD ########################################################

    #
    # data = torch.reshape(torch.from_numpy(data), [-1, 32, 32])
    # # [np.reshape(np.rint(127.5 * (clusters[s] + 1.0)), [32, 32, 3]).astype(np.uint8) for s in samples]
    #
    # sample_new = torch.reshape(torch.round(127.5 * (torch.index_select(clusters, 0, data.long().flatten()) + 1.0)),
    #                        [data.shape[0], 3, 32, 32]).to('cuda')
    # sample = torch.reshape(torch.round(127.5 * (clusters[data.long()] + 1.0)), [data.shape[0], 3, 32, 32]).to('cuda')
    # yos = torch.eq(sample,sample_new)
    # import ipdb; ipdb.set_trace()

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
    npgrid = npgrid.astype(np.uint8)
    im = np.transpose(npgrid, (1, 2, 0))
    plt.imsave(export_img,im)


data_path = './cifar10'
(trX, trY), (vaX, vaY), (teX, teY) = load_data(data_path)
sample = transform_cluster_to_image(vaX)

log_dir = '/home/dsi/eyalbetzalel/pytorch-generative-v6/image_test/test1.png'
# _summary_writer = tensorboard.SummaryWriter(log_dir = log_dir, max_queue=100)
# _summary_writer.add_images('sample',sample,0)
# _summary_writer.close()

pytorch_tensor = sample[1:48,:,:,:]
plot_images_grid(pytorch_tensor, export_img=log_dir)
import ipdb; ipdb.set_trace()





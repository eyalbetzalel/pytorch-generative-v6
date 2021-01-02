import numpy as np
import os
import torch
from torch.utils import tensorboard
import torchvision
import matplotlib.pyplot as plt
import h5py



pathToCluster = r"/home/dsi/eyalbetzalel/image-gpt/downloads/kmeans_centers.npy"  # TODO : add path to cluster dir
global clusters
clusters = torch.from_numpy(np.load(pathToCluster)).float()

def transform_cluster_to_image(samples):

    data_tor = torch.reshape(torch.from_numpy(samples), [-1, 32, 32])
    clusters = torch.from_numpy(np.load(pathToCluster)).float()
    sample_new = torch.round(127.5 * (clusters[data_tor.long()] + 1.0))
    sample_new = sample_new.permute(0, 3, 1, 2)

    return sample_new

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


# data_path = './cifar10'
# (trX, trY), (vaX, vaY), (teX, teY) = load_data(data_path)
# sample = transform_cluster_to_image(vaX)
# log_dir = '/home/dsi/eyalbetzalel/pytorch-generative-v6/image_test/test1.png'

directory = "./"
train, test = load_h5_dataset(directory)
sample = transform_cluster_to_image(train)
log_dir = '/home/dsi/eyalbetzalel/pytorch-generative-v6/image_test'

pytorch_tensor = sample[1:48,:,:,:]
plot_images_grid(pytorch_tensor, export_img=log_dir)






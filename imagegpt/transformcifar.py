import numpy as np
from pytorch-generative-v6.image_gpt_as_photo import transform_cluster_to_image

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
from __future__ import division, print_function


import numpy as np
import torch
from torch.utils.data import Dataset


def load_mnist(path='./data/per.csv'):
    # with open(r"F:\IDEC\data\ucecec_go.csv", 'r') as f:
    x = pd.read_csv(f)
    #f = np.load(path)
    #x = f
    # print('samples', x.shape)
    # print(x.shape)
    return x
#
# def load_mnist(path='./data/mnist.npz'):
#     f = np.load(path)
#     x = f
#     print('MNIST samples', x.shape)
#     return x
# -*- coding: utf-8 -*-
#
# Copyright © dawnranger.
#
# 2018-05-08 10:15 <dawnranger123@gmail.com>
#
# Distributed under terms of the MIT license.
from __future__ import division, print_function
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
seed = 1
import random
import math
from math import log
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)




# def load_mnist(path='./data/mnist.npz'):
#     f = np.load(path)
#     x_train, x_test = f['x_train'],  f['x_test']
#     f.close()
#     x = x_test#np.concatenate((x_train, x_test))
#     x = x.reshape((x.shape[0], -1)).astype(np.float32)
#     x = np.divide(x, 255.)
#     print('MNIST samples', x.shape)
#     print(type(x))
#     return x

def load_mnist(path='./sep_1.csv'):
    # f = pd.read_csv(r'C:\Users\ch\Desktop\ccidec\ccidec\data\ucec.csv')
    with open(path) as f:
        x = pd.read_csv(f,index_col=0).transpose().astype(np.float32)
    f.close()
    x = np.array(x)
    print('samples', x.shape)
    x = x[:,3:]
    # x = log((x+1), 2)  #####这里x所有数据预处理 转为log2（X+1)
    print('samples', x.shape)
    return x


class MnistDataset(Dataset):

    def __init__(self):
        # self.x, self.y = load_mnist()
        self.x = load_mnist()

    def __len__(self):
        return self.x.shape[0]

    # def __getitem__(self, idx):
    #     return torch.from_numpy(np.array(self.x[idx])), torch.from_numpy(
    #         np.array(self.y[idx])), torch.from_numpy(np.array(idx))

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])), torch.from_numpy(np.array(idx))

#######################################################
# Evaluate Critiron
#######################################################


# def cluster_acc(y_true, y_pred):
#     """
#     Calculate clustering accuracy. Require scikit-learn installed
#
#     # Arguments
#         y: true labels, numpy.array with shape `(n_samples,)`
#         y_pred: predicted labels, numpy.array with shape `(n_samples,)`
#
#     # Return
#         accuracy, in [0,1]
#     """
#     y_true = y_true.astype(np.int64)
#     assert y_pred.size == y_true.size
#     D = max(y_pred.max(), y_true.max()) + 1
#     w = np.zeros((D, D), dtype=np.int64)
#     for i in range(y_pred.size):
#         w[y_pred[i], y_true[i]] += 1
#     from sklearn.utils.linear_assignment_ import linear_assignment
#     ind = linear_assignment(w.max() - w)
#     return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

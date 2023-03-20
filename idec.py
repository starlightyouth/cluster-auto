# -*- coding: utf-8 -*-
#
# Copyright © dawnranger.
#
# 2018-05-08 10:15 <dawnranger123@gmail.com>
#
# Distributed under terms of the MIT license.
from __future__ import print_function, division
import argparse
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
import torch

import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn import Linear
import matplotlib.pyplot as plt
#from sklearn import metrics
from sklearn.metrics import silhouette_score, davies_bouldin_score
from utils import MnistDataset
seed = 1   #种子
import torch.backends.cudnn as cudnn
cudnn.deterministic = True
cudnn.benchmark = True
import random
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)


class AE(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z):
        super(AE, self).__init__()

        # encoder
        self.enc_1 = Linear(n_input, n_enc_1)
        self.enc_2 = Linear(n_enc_1, n_enc_2)
       # self.enc_3 = Linear(n_enc_2, n_enc_3)

        self.z_layer = Linear(n_enc_2, n_z)

        # decoder
        # self.dec_1 = Linear(n_z, n_dec_1)
        self.dec_2 = Linear(n_z, n_dec_2)
        self.dec_3 = Linear(n_dec_2, n_dec_3)

        self.x_bar_layer = Linear(n_dec_3, n_input)

    def forward(self, x):

        # encoder
        x = x + torch.randn(x.size())*0.1
        enc_h1 = F.relu(self.enc_1(x))
        enc_h2 = F.relu(self.enc_2(enc_h1))
       # enc_h3 = F.relu(self.enc_3(enc_h2))
        z = self.z_layer(enc_h2)  #降维数据
        # decoder
       # dec_h1 = F.relu(self.dec_1(z))
        dec_h2 = F.relu(self.dec_2(z))
        dec_h3 = F.relu(self.dec_3(dec_h2))
        x_bar = self.x_bar_layer(dec_h3)   #重构数据

        return x_bar, z   #返回降维数据和重构数据



class IDEC(nn.Module):

    def __init__(self,
                 n_enc_1,
                 n_enc_2,
                 n_enc_3,
                 n_dec_1,
                 n_dec_2,
                 n_dec_3,
                 n_input,
                 n_z,
                 n_clusters,   #聚类中心个数
                 alpha=1,      #k-means算法的超参数alpha
                 pretrain_path='data/ae_pre.pkl'):  #表示自编码器预训练的权重路径
        super(IDEC, self).__init__()
        self.alpha = 1.0
        self.pretrain_path = pretrain_path  #预训练路径

        self.ae = AE(
            n_enc_1=n_enc_1,
            n_enc_2=n_enc_2,
            n_enc_3=n_enc_3,
            n_dec_1=n_dec_1,
            n_dec_2=n_dec_2,
            n_dec_3=n_dec_3,
            n_input=n_input,
            n_z=n_z)
        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_z))   #n_clusters表示期望聚类的数量，n_z表示编码器输出的特征向量的维数
        torch.nn.init.xavier_normal_(self.cluster_layer.data)  #随机初始化self.cluster_layer中的元素

    def pretrain(self, path=''):   #预训练,如果 path 参数为空，则调用 pretrain_ae 函数对 AE 进行预训练。如果 path 参数不为空，则会从指定路径加载预训练模型的权重。加载完成后，打印一条日志信息表示已经完成预训练。
        if path == '':
            pretrain_ae(self.ae)
        # load pretrain weights
        self.ae.load_state_dict(torch.load(self.pretrain_path))
        print('load pretrained ae from', path)

    def forward(self, x):

        x_bar, z = self.ae(x)
        # cluster
        q = 1.0 / (1.0 + torch.sum(
            torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.alpha)
        q = q.pow((self.alpha + 1.0) / 2.0)   #这一行代码对概率q进行了指数平滑的处理，以确保概率值具有更强的表现力
        q = (q.t() / torch.sum(q, 1)).t()   #对概率值q进行标准化处理，确保每个样本点的概率值之和为1
        return x_bar, q   #重构值x_bar和聚类概率q，用于计算聚类损失和重构损失


def target_distribution(q):    #q 聚类概率,返回目标分布矩阵。目标分布矩阵是一个与聚类簇数目相同的矩阵，用于指导EM算法的聚类过程
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()  #.t()转置


def pretrain_ae(model):
    '''
    pretrain autoencoder  预训练自编码器
    '''
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    print("??")
    print(model)
    optimizer = Adam(model.parameters(), lr=args.lr)
    for epoch in range(args.pretrain_epochs):
        total_loss = 0.
        for batch_idx, (x, _) in enumerate(train_loader):  # x 是输入数据，_ 是对应的标签（因为这里不需要用到标签，所以用 _ 占位），因为是无监督学习
            x = x.to(device)

            optimizer.zero_grad()
            x_bar, z = model(x)
            loss = F.mse_loss(x_bar, x)
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        print("epoch {} loss={:.4f}".format(epoch,
                                            total_loss / (batch_idx + 1)))
        torch.save(model.state_dict(), args.pretrain_path)
    print("model saved to {}.".format(args.pretrain_path))


def train_idec():

    model = IDEC(
        n_enc_1=500,
        n_enc_2=100,
        n_enc_3=20,
        n_dec_1=20,
        n_dec_2=100,
        n_dec_3=500,
        n_input=args.n_input,
        n_z=args.n_z,
        n_clusters=args.n_clusters,
        alpha=1.0,
        pretrain_path=args.pretrain_path).to(device)

    # model.pretrain('data/ae_pre.pkl') #直接读取保存的模型
    model.pretrain()  #  迁移学习

    train_loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False)
    optimizer = Adam(model.parameters(), lr=args.lr)

    # cluster parameter initiate
    data = dataset.x
    # y = dataset.y
    data = torch.Tensor(data).to(device)
    x_bar, hidden = model.ae(data)

    kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)
    y_pred = kmeans.fit_predict(hidden.data.cpu().numpy())
    silhouetteScore = silhouette_score(hidden.data.cpu().numpy(), y_pred, metric='euclidean')
    davies_bouldinScore = davies_bouldin_score(hidden.data.cpu().numpy(), y_pred)

    print("silhouetteScore={:.4f}".format(silhouetteScore),', davies_bouldinScore {:.4f}'.format(davies_bouldinScore))

    hidden = None
    x_bar = None

    y_pred_last = y_pred
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)

    model.train()
    for epoch in range(args.train_epochs):

        if epoch % args.update_interval == 0:

            _, tmp_q = model(data)
            _, hidden = model.ae(data)
            # update target distribution p
            tmp_q = tmp_q.data
            p = target_distribution(tmp_q)

            # evaluate clustering performance
            y_pred = tmp_q.cpu().numpy().argmax(1)
            delta_label = np.sum(y_pred != y_pred_last).astype(
                np.float32) / y_pred.shape[0]
            y_pred_last = y_pred
            silhouetteScore = silhouette_score(hidden.data.cpu().numpy(), y_pred, metric='euclidean')
            davies_bouldinScore = davies_bouldin_score(hidden.data.cpu().numpy(), y_pred)
            print('Iter {}'.format(epoch), ':silhouetteScore {:.4f}'.format(silhouetteScore),
                  ', davies_bouldinScore {:.4f}'.format(davies_bouldinScore),', delta_label {:.4f}'.format(delta_label))#, ', calinski_harabaszScore {:.4f}'.format(calinski_harabaszScore))
            #print(y_pred.shape)
            # np.savetxt(r'x.csv', y_pred, delimiter=',')
            #np.savetxt(r'C:\pypro\x_x.csv', hidden.data.cpu().numpy(), delimiter=',')
            d = np.column_stack((hidden.data.cpu().numpy(), y_pred))
            # print(d.shape)
            np.savetxt(r'x_tsne.csv', d, delimiter=',')
            # x1 = d[d[:, 2]==0]
            # x2 = d[d[:, 2]==1]
            # x3 = d[d[:, 2]==2]
            # x4 = d[d[:, 2]==3]
            # x5 = d[d[:, 2]==4]
            # x6 = d[d[:, 2]==5]
            # x7 = d[d[:, 2]==6]
            # x8 = d[d[:, 2]==7]
            # x9 = d[d[:, 2]==8]
            # # print(x3.shape)
            # #plt.scatter(hidden.data.cpu().numpy()[:, 0], hidden.data.cpu().numpy()[:, 1], c="red", marker='o', label='see')
            # plt.scatter(x1[:, 0], x1[:, 1], c="red", marker='o', label='label0')
            # plt.scatter(x2[:, 0], x2[:, 1], c="green", marker='*', label='label1')
            # plt.scatter(x3[:, 0], x3[:, 1], c="blue", marker='+', label='label2')
            # plt.scatter(x4[:, 0], x4[:, 1], c="yellow", marker='o', label='label0')
            # plt.scatter(x5[:, 0], x5[:, 1], c="purple", marker='*', label='label1')
            # plt.scatter(x6[:, 0], x6[:, 1], c="brown", marker='+', label='label2')
            # plt.scatter(x7[:, 0], x7[:, 1], c="pink", marker='o', label='label0')
            # plt.scatter(x8[:, 0], x8[:, 1], c="yellowgreen", marker='*', label='label1')
            # plt.scatter(x9[:, 0], x9[:, 1], c="skyblue", marker='+', label='label2')
            # plt.xlabel('petal length')
            # plt.ylabel('petal width')
            # plt.legend(loc=2)
            # plt.savefig('brca.png')
            # plt.show()

            #print(y_pred)

            #
            if epoch > 0 and delta_label < args.tol:
                print('delta_label {:.4f}'.format(delta_label), '< tol',
                      args.tol)
                print('Reached tolerance threshold. Stopping training.')
                break
        for batch_idx, (x, idx) in enumerate(train_loader):
            # print(x.shape)
            # print(torch.randn(x.shape).shape)
            x = x.to(device)
            x1 = (x + 0.01 * torch.randn(x.shape)).to(device)   #added nosie
            # x1 = x.to(device)
            idx = idx.to(device)
            idx = idx.long()
            x_bar, q = model(x1)

            reconstr_loss = F.mse_loss(x_bar, x)
            kl_loss = F.kl_div(q, p[idx])
            loss =  kl_loss + args.gamma * reconstr_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--n_clusters', default=7, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--n_z', default=2, type=int)
    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--pretrain_path', type=str, default='data/ae_pre')
    parser.add_argument(
        '--gamma',
        default=0.5,
        type=float,
        help='coefficient of clustering loss')
    parser.add_argument('--update_interval', default=1, type=int)
    parser.add_argument('--tol', default=0.001, type=float)
    parser.add_argument('--pretrain_epochs', default=50, type=int)
    parser.add_argument('--train_epochs', default=500, type=int)
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    # print("use cuda: {}".format(args.cudla))
    device = torch.device("cuda" if args.cuda else "cpu")

    if args.dataset == 'mnist':
        args.pretrain_path = 'data/ae_pre.pkl'
        args.n_clusters = 2
        dataset = MnistDataset()
        args.pretrain_epochs = 10  #(2,5,10,20)
        args.train_epochs = 15    #(5,10,20,40,50)
        args.n_z = 50             #(5,10,20,50)
        args.gamma = 0.1            #(1,0.1,0.5,2,10)
        args.n_input = dataset.x.shape[1]
        args.tol = 0#.05#
        args.lr = 0.001           #(0.01,0.001,0.0001)
        args.batch_size = 512      #8,16,32
    print(args)
    #print(dataset.x.shape)



    # np.savetxt(r'x.csv', dataset.x, delimiter=',')
    train_idec()


import matplotlib.pyplot as plt

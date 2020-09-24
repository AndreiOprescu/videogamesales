# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler

# Extracting the data
dataset = pd.read_csv("vgsales.csv")
X = dataset.iloc[:, 2:].values

def denullify(rows, array):
    for row in rows:
        if type(array[1, row]) == float:
            array[:, row][pd.isnull(array[:, row])] = 0
        if type(array[1, row]) == int:
            array[:, row][pd.isnull(array[:, row])] = 0
        if type(array[1, row]) == str:
            array[:, row][pd.isnull(array[:, row])] = "NaN"
        if type(array[1, row]) == type(None):
            for val in array[:, row]:
                if type(array[1, row]) == int:
                    array[:, row][pd.isnull(array[:, row])] = 0
                    continue
                if type(array[1, row]) == str:
                    array[:, row][pd.isnull(array[:, row])] = "NaN"
                    continue

denullify(range(X.shape[1]), X)

# Encoding categorical data
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
for r in range(4):
    le = LabelEncoder()
    X[:, r] = le.fit_transform(X[:, r])

encoder = OneHotEncoder(categorical_features=[range(4)])
X = encoder.fit_transform(X).toarray()

# Scaling the data
sc = MinMaxScaler()
X = sc.fit_transform(X)

# getting the number of games
n_games = len(X)

# Getting the number of features
n_features = X.shape[1]

# Making the train and test set (80% train, 20% test)
train_set = X[:int(n_games*0.8)]
test_set = X[int(n_games*0.8):]

# Making the data to Torch tensors

train_set = torch.FloatTensor(train_set)
test_set = torch.FloatTensor(test_set)

class RBM(nn.Module):
    def __init__(self, nv, nh):
        self.W = torch.randn(nh, nv)
        self.a = torch.randn(1, nh)
        self.b = torch.randn(1, nv)
    
    def sample_h(self, x):
        wx = torch.mm(x, self.W.t())
        activation = wx + self.a.expand_as(wx)
        p_h_given_v = torch.sigmoid(activation)
        return p_h_given_v, torch.bernoulli(p_h_given_v)
    
    def sample_v(self, y):
        wy = torch.mm(y, self.W)
        activation = wy + self.b.expand_as(wy)
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)
    
    def train(self, v0, vk, ph0, phk):
        self.W += (torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)).t()
        self.a += torch.sum((ph0 - phk), 0)
        self.b += torch.sum((v0 - vk), 0)

nv = train_set.shape[1]
nh = 100
batch_size = 100
rbm = RBM(nv, nh)

n_epochs = 10
for epoch in range(1, n_epochs + 1):
    train_loss = 0
    s = 0.
    for batch in range(0, n_games - batch_size, batch_size):
        v0 = train_set[batch:batch+batch_size]
        vk = train_set[batch:batch+batch_size]
        ph0, _ = rbm.sample_h(v0)
        for _ in range(10):
            _, hk = rbm.sample_h(vk)
            _, vk = rbm.sample_v(hk)
            vk[v0<0] = v0[v0<0]
        phk, _ = rbm.sample_h(vk)
        rbm.train(v0, vk, ph0, phk)
        train_loss += torch.mean(torch.abs(v0[v0>=0] - vk[v0>=0]))
        s += 1.
    print('epoch: ' + str(epoch) + ' loss: ' + str(train_loss/s))



train_loss = 0
s = 0.
for game in range(n_games):
    v = train_set[game:game+1]
    vt = test_set[game:game+1]
    ph0, _ = rbm.sample_h(v0)
    if len(vt[vt>=0]) > 0
        _, h = rbm.sample_h(v)
        _, v = rbm.sample_v(h)
        train_loss += torch.mean(torch.abs(v0[v0>=0] - vk[v0>=0]))
        s += 1.
print('Test loss: ' + str(train_loss/s))
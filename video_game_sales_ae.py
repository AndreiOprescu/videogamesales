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

#Creating the Autoencoder class
class SAE(nn.Module):
    def __init__(self):
        super(SAE, self).__init__()
        self.fc1 = nn.Linear(n_features, 20)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, 20)
        self.fc4 = nn.Linear(20, n_features)
        self.activation = nn.Sigmoid()
    
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.activation(self.fc4(x))
        return x

sae = SAE()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(sae.parameters(), lr = 0.01, weight_decay = 0.5)

# Training the autoencoder
nb_epoch = 100
for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    s = 0.
    for game in range(n_games):
        input = Variable(train_set[game]).unsqueeze(0)
        target = input.clone()
        if torch.sum(target.data > 0) > 0:
            output = sae.forward(input)
            target.require_grad = False
            output[target == 0] = 0
            loss = criterion(output, target)
            mean_corrector = n_games/float(torch.sum(target.data > 0) + 1e-10)
            loss.backward()
            train_loss += np.sqrt(loss.item()*mean_corrector)
            s += 1.
            optimizer.step()
    print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))
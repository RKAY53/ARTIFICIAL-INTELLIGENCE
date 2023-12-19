# neuralnet.py
#Kalaipriyan R
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/29/2019
# Modified by James Soole for the Fall 2023 semester

"""
This is the main entry point for MP9. You should only modify code within this file.
The unrevised staff files will be used for all other files and classes when code is run, 
so be careful to not modify anything else.
"""


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class NeuralNet(nn.Module):
    def __init__(self, in_size=2883, h=150, out_size=4, lrate=0.01):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(in_size, h)
        self.fc2 = nn.Linear(h, out_size)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=lrate)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))  #sigmoid activation function
        x = self.fc2(x) 
        return x

    def step(self, x, y):
        self.train() 
        self.optimizer.zero_grad() 
        output = self.forward(x)  
        loss = self.loss_fn(output, y)  # Compute loss
        loss.backward()  
        self.optimizer.step() 
        return loss.item()  

def standardize_data(X):
    """Standardize the dataset"""
    X = X.astype(np.float32)
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    X_standardized = (X - mu) / sigma
    return torch.tensor(X_standardized, dtype=torch.float)

def fit(train_set, train_labels, dev_set, epochs, batch_size=100):
    """
    Train and evaluate a neural network.
    """
    train_set = standardize_data(train_set.detach().numpy())
    dev_set = standardize_data(dev_set.detach().numpy())

    train_labels = train_labels.long()
    net = NeuralNet()

    #DataLoader
    train_tensor = TensorDataset(train_set, train_labels)
    train_loader = DataLoader(train_tensor, batch_size=batch_size, shuffle=True)

    dev_tensor = TensorDataset(dev_set, torch.zeros(len(dev_set), dtype=torch.long))
    dev_loader = DataLoader(dev_tensor, batch_size=batch_size)

    losses = []

    for epoch in range(epochs):
        net.train()
        total_loss = 0
        for data, target in train_loader:
            loss = net.step(data, target)
            total_loss += loss
        losses.append(total_loss / len(train_loader))

    net.eval()
    yhats = []
    with torch.no_grad():
        for data, _ in dev_loader:
            output = net(data)
            predicted = torch.max(output, 1)[1]
            yhats.extend(predicted.detach().cpu().numpy())
    yhats = np.array(yhats, dtype=np.int64).astype(int)
    return losses, yhats, net
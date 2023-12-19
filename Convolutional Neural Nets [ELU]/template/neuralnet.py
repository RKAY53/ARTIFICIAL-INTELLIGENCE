# neuralnet.py
# Kalaipriyan R
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/29/2019
# Modified by James Soole for the Fall 2023 semester

"""
This is the main entry point for MP10. You should only modify code within this file.
The unrevised staff files will be used for all other files and classes when code is run, 
so be careful to not modify anything else.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class NeuralNet(nn.Module):
    def __init__(self, h=150, out_size=4, lrate=0.01):
        super(NeuralNet, self).__init__()
        
        # CNN Layers
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), 
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.flattened_size = 64 * 7 * 7  
        self.fc1 = nn.Linear(self.flattened_size, h)
        
        self.activation = nn.ELU()         # ELU Activation Function

        self.fc2 = nn.Linear(h, out_size)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=lrate, weight_decay=1e-5)

    def forward(self, x):
        if x.dim() == 2:
            x = x.view(-1, 3, 31, 31)  
        
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def step(self, x, y):
        self.train()
        self.optimizer.zero_grad()
        output = self.forward(x)
        loss = self.loss_fn(output, y)
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

def fit(train_set, train_labels, dev_set, epochs, batch_size=100, lrate=0.01):
    """
    Train and evaluate a neural network.
    """
    train_set = standardize_data(train_set.detach().numpy())
    dev_set = standardize_data(dev_set.detach().numpy())

    if not isinstance(train_labels, torch.Tensor):
        train_labels = torch.tensor(train_labels, dtype=torch.long)
    
    net = NeuralNet(h=150, out_size=4, lrate=lrate)
    train_tensor = TensorDataset(train_set, train_labels)
    train_loader = DataLoader(train_tensor, batch_size=batch_size, shuffle=True)       # DataLoader for training data

    # Training loop
    losses = []
    for epoch_num in range(epochs):
        net.train()
        total_loss = 0
        for data, target in train_loader:
            net.optimizer.zero_grad()
            output = net(data)
            loss = net.loss_fn(output, target)
            loss.backward()
            net.optimizer.step()
            total_loss += loss.item()
        average_loss = total_loss / len(train_loader)
        losses.append(average_loss)
        print(f'Epoch {epoch_num+1}/{epochs}, Training Loss: {average_loss:.4f}')

    net.eval()
    dev_predictions = []
    with torch.no_grad():
        for data in dev_set:
            output = net(data.unsqueeze(0))           # Add batch dimension
            predicted = torch.max(output, 1)[1]
            dev_predictions.append(predicted.item())

    dev_predictions = np.array(dev_predictions, dtype=np.int64)

    return losses, dev_predictions, net
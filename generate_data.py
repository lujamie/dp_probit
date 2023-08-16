'''
generates synthetic data with the following constraints:
1. X ~ Unif(-1, 1)
2. betas ~ Normal(0, sqrt(N))
3. Y = 0 if X*beta <= 0, Y = 1 if X*beta > 0 for arbitrary beta
'''

import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data.clone().detach().float()
        self.labels = labels.clone().detach().float()
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = {
            'data': self.data[idx],
            'label': self.labels[idx]
        }
        return sample


def torch_data(X, y):
    bat_size = 1
    X_tensor = torch.from_numpy(X)
    y_tensor = torch.from_numpy(y)
    TD = CustomDataset(X_tensor, y_tensor)
    data_loader = DataLoader(TD, batch_size=bat_size, shuffle=True)
    return data_loader

def generate_x(N, K):
    X = np.empty(shape=(N, K))
    for i in range(N):
        X[i] = np.random.uniform(low=-1, high=1, size=K)
    return X

def generate(N, K, beta):
    X = generate_x(N, K)
    y = np.empty(shape=N)
    for i in range(N):
        rho = np.random.standard_normal()
        xbeta = np.dot(X[i], beta)
        if xbeta + rho > 0:
            y[i] = 1
        else:
            y[i] = 0
    return X, y

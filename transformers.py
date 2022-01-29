#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 11:52:21 2022

@author: nvakili
"""

import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision

class Wine(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        xy = np.loadtxt('./data/wine.csv', delimiter=',', dtype=np.float32, skiprows=1)
        self.n_samples = xy.shape[0]

        self.x_data = xy[:, 1:]
        self.y_data = xy[:, [0]]
    
    def __getitem__(self, ind):
        sample = self.x_data[ind], self.y_data[ind]   
        if self.transform:
            sample = self.transform(sample)
        return sample
    
    def __len__(self):
        return self.n_samples
    
class ToTensor:
    def __call__(self, sample):
        x,y = sample
        return torch.from_numpy(x), torch.from_numpy(y)
    
class MulTransform:
    def __init__(self, factor):
        self.factor = factor
    
    def __call__(self, sample):
        x, y = sample
        x *= self.factor
        return x, y 
    
print('without transform:')
data = Wine()
first_data = data[0]
features, labels = first_data
print(type(features), type(labels))
print(features, labels)

print('\nWith Tensor Transform')
dataset = Wine(transform=ToTensor())
first_data = dataset[0]
features, labels = first_data
print(type(features), type(labels))
print(features, labels)

print('\nWith Tensor and Multiplication Transform')
composed = torchvision.transforms.Compose([ToTensor(), MulTransform(4)])
dataset = Wine(transform=composed)
first_data = dataset[0]
features, labels = first_data
print(type(features), type(labels))
print(features, labels)

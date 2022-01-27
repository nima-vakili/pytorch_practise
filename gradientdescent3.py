#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 11:20:17 2022

@author: nvakili
"""

import torch
import torch.nn as nn

X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)

n_samples, n_features = X.shape
X_test = torch.tensor([5], dtype=torch.float32)

input_size = n_features
output_size = n_features

model = nn.Linear(input_size, output_size)
print(f'prediction before training: f(5):{model(X_test).item():.3f}')

learning_rate = .01
n_iters = 100

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(n_iters):
    y_pred = model(X)
    l = loss(y, y_pred)
    l.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    if epoch % 10 == 0:
        [w, b] = model.parameters() # unpack parameters
        print('epoch ', epoch+1, ': w = ', w[0][0].item(), ' loss = ', l)

print(f'Prediction after training: f(5) = {model(X_test).item():.3f}')
















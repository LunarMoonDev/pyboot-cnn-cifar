# importing libraries
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid

from model import ConvolutionalNetwork

torch.manual_seed(101)

# constants
DATA = './data'
VERSION = '0.0'

# data prep
transform = transforms.ToTensor()

train_data = datasets.CIFAR10(root = DATA, train = True, download = True, transform = transform)
test_data = datasets.CIFAR10(root = DATA, train = False, download = True, transform = transform)

train_loader = DataLoader(train_data, batch_size = 10, shuffle = True)
test_loader = DataLoader(test_data, batch_size = 10, shuffle = False)

# prep model
model = ConvolutionalNetwork()

# prep loss and optim
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)


start_time = time.time()

epochs = 10
train_losses = []
test_losses = []
train_corr = []
test_corr = []

for i in range(epochs):
    trn_corr = 0
    tst_corr = 0

    for b, (X_train, y_train) in enumerate(train_loader):
        b += 1

        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)

        # tally the correct ones
        predicted = torch.max(y_pred.data, 1)[1]
        batch_corr = (predicted == y_train).sum()
        trn_corr += batch_corr

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if b%1000 == 0:
            print(f'epoch: {i:2} batch: {b:4} [{10 * b:6}/50000] loss: {loss.item():10.8f} \ acc: {trn_corr.item() * 100 / (10 * b): 7.3}%')
    
    train_losses.append(loss.item())   # get the last loss
    train_corr.append(trn_corr)

    # running test data for analysis
    with torch.no_grad():
        for b, (X_test, y_test) in enumerate(test_loader):

            y_val = model(X_test)

            # tally the correct ones
            predicted = torch.max(y_val.data, 1)[1]
            tst_corr += (predicted == y_test).sum()
    
    loss = criterion(y_val, y_test) # get the last loss
    test_losses.append(loss.item())
    test_corr.append(tst_corr)

print(f'\nDuration: {time.time() - start_time:.0f} seconds')

# saving resources
torch.save(model.state_dict(),  f'./models/model.S.{VERSION}.{int(time.time())}.pt')
pd.DataFrame(np.array(train_losses)).to_csv('./data/interim/train_losses.csv', header = None, index = False)
pd.DataFrame(np.array(test_losses)).to_csv('./data/interim/test_losses.csv', header = None, index = False)
pd.DataFrame(np.array(train_corr)).to_csv('./data/interim/train_corr.csv', header = None, index = False)
pd.DataFrame(np.array(test_corr)).to_csv('./data/interim/test_corr.csv', header = None, index = False)

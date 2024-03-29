# importing libraries
import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy import genfromtxt

# constants
TRAIN_LOSSES = './data/interim/train_losses.csv'
TEST_LOSSES = './data/interim/test_losses.csv'
TRAIN_CORR = './data/interim/train_corr.csv'
TEST_CORR = './data/interim/test_corr.csv'

# prep the data
train_losses = genfromtxt(TRAIN_LOSSES, delimiter = ',', dtype = np.float)
test_losses = genfromtxt(TEST_LOSSES, delimiter = ',', dtype = np.float)
train_corr = genfromtxt(TRAIN_CORR, delimiter = ',', dtype = np.float)
test_corr = genfromtxt(TEST_CORR, delimiter = ',', dtype = np.float)

train_losses = torch.tensor(train_losses, dtype = torch.float)
test_losses = torch.tensor(test_losses, dtype = torch.float)
train_corr = torch.tensor(train_corr, dtype = torch.int)
test_corr = torch.tensor(test_corr, dtype = torch.int)

# plots
plt.plot(train_losses.tolist(), label = 'training loss')
plt.plot(test_losses.tolist(), label = 'validation losses')
plt.title('Loss at the end of each epoch')
plt.legend()
plt.savefig('./reports/figures/loss_epoch.png')

plt.close()
plt.plot([t / 500 for t in train_corr], label = 'training accuracy')
plt.plot([t / 100 for t in test_corr], label = 'validation accuracy')
plt.title('Accuracy at the end of each epoch')
plt.legend()
plt.savefig('./reports/figures/acc_epoch.png')

print(test_corr)
print()
print(f'Test accuracy: {test_corr[-1].item() * 100/ 10000: .3f}%')

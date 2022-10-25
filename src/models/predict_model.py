# importing libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid

from model import ConvolutionalNetwork

torch.manual_seed(101)

# constants
DATA = './data'
MODEL_NAME = './models/model.S.0.0.1666611137.pt'

# prep data
transform = transforms.ToTensor()

test_data = datasets.CIFAR10(root = DATA, train = False, download = True, transform = transform)
test_load_all = DataLoader(test_data, batch_size = 10000, shuffle = False)

# prep model
model = ConvolutionalNetwork()
model.load_state_dict(torch.load(MODEL_NAME))
model.eval()

# prep criterion
criterion = nn.CrossEntropyLoss()

# prediction
class_names = ['plane', '  car', ' bird', '  cat', ' deer', '  dog', ' frog', 'horse', ' ship', 'truck']

with torch.no_grad():
    correct = 0
    for X_test, y_test in test_load_all:
        y_val = model(X_test)
        predicted = torch.max(y_val, 1)[1]
        correct += (predicted == y_test).sum()

arr = confusion_matrix(y_test.view(-1), predicted.view(-1))
df_cm = pd.DataFrame(arr, class_names, class_names)
plt.figure(figsize = (9, 6))
sn.heatmap(df_cm, annot = True, fmt = "d", cmap = 'BuGn')
plt.xlabel("prediction")
plt.ylabel("label (ground truth)")
plt.savefig('./reports/figures/heatmap.png')

misses = np.array([])
for i in range(len(predicted.view(-1))):
    if predicted[i] != y_test[i]:
        misses = np.append(misses, i).astype('int64')

r = 8
row = iter(np.array_split(misses, len(misses) // r + 1))

np.set_printoptions(formatter = dict(int = lambda x: f'{x: 5}'))

nextrow = next(row)
lbls = y_test.index_select(0, torch.tensor(nextrow)).numpy()
gues = predicted.index_select(0, torch.tensor(nextrow)).numpy()

print("Index: ", nextrow)
print("Label: ", lbls)
print("Class: ", *np.array([class_names[i] for i in lbls]))
print()
print("Guess: ", gues)
print("Class: ", *np.array([class_names[i] for i in gues]))


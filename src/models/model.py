# importing libraries
import torch.nn as nn
import torch.nn.functional as F


class ConvolutionalNetwork(nn.Module):
    '''
        basic model for implementing a CNN model with hardcoded parameters
        it has two conv layers with max pooling
    '''

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 3, 1)
        self.conv2 = nn.Conv2d(6, 16, 3, 1)
        self.fc1 = nn.Linear(6 * 6 * 16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, X):
        '''
            connects the layers in __init__ with given input X

            @param X, 32 x 32 x 3 image
        '''
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2, 2)
        X = X.view(-1, 6 * 6 * 16)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)

        return F.log_softmax(X, dim = 1)

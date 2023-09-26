import torch
from torch import nn
import torch.nn.functional as F


class autoencoderMLP4Layer(nn.Module):

    def __init__(self, N_inputs=784, N_bottlenecks=8, N_outputs=784):
        super(autoencoderMLP4Layer, self).__init__()
        N2 = int(N_inputs/2)
        self.fc1 = nn.Linear(N_inputs, N2)
        self.fc2 = nn.Linear(N2, N_bottlenecks)
        self.fc3 = nn.Linear(N_bottlenecks, N2)
        self.fc4 = nn.Linear(N2, N_outputs)

        self.type = 'MLP4'
        self.input_shape = (1,28*28)
    def encoder(self, X):
        X = self.fc1(X)
        X = F.relu(X)
        X = self.fc2(X)
        X = F.relu(X)
        return X

    def decoder(self, X):
        X = self.fc3(X)
        X = F.relu(X)
        X = self.fc4(X)
        X = torch.sigmoid(X)
        return X

    def forward(self, X):
        X = self.encoder(X)
        X = self.decoder(X)

        return X
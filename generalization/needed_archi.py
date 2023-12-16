import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.nn import functional as F

class MaxoutLayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_units):
        super(MaxoutLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_units = num_units
        self.fc = nn.Linear(input_dim, output_dim * num_units)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, self.output_dim, self.num_units)
        x, _ = torch.max(x, dim=2)
        return x


class DeepMaxoutNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_units, num_layers):
        super(DeepMaxoutNetwork, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_units = num_units
        self.num_layers = num_layers

        self.layers = nn.ModuleList()
        self.layers.append(MaxoutLayer(input_dim, hidden_dim, num_units))

        for _ in range(num_layers - 2):
            self.layers.append(MaxoutLayer(hidden_dim, hidden_dim, num_units))

        self.layers.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        return x

class ShallowSoftmaxNetBN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ShallowSoftmaxNetBN, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.bn = nn.BatchNorm1d(output_dim)  # Batch Normalization
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)  # Apply Batch Normalization
        x = self.softmax(x)
        return x
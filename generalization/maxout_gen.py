import torch
import torch.nn as nn
import math
import numpy as np

"""
    On va d'abord implémenter une fonction d'activation Maxout : Maxout(x) = max(x1, x2, ..., xk)
    Class maxout récupéré sur le git https://github.com/thetechdude124/Custom-Maxout-Activation-Implementation/tree/master

"""

# Define the Maxout layer
# Maxout Layer
class MaxoutLayer(nn.Module):
    def __init__(self, input_size, output_size, num_pieces=2):
        super(MaxoutLayer, self).__init__()
        self.fc = nn.Linear(input_size, output_size * num_pieces)
        self.num_pieces = num_pieces

    def forward(self, x):
        x = self.fc(x)
        x, _ = x.view(x.size(0), -1, self.num_pieces).max(-1)
        return x

# Maxout Network with Softmax Layer
class MaxoutNetworkWithSoftmax(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MaxoutNetworkWithSoftmax, self).__init__()
        self.layer1 = MaxoutLayer(input_size, 512)
        self.layer2 = MaxoutLayer(512, 256)
        self.softmax = nn.Softmax(dim=1)
        self.output_layer = nn.Linear(256, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.layer1(x))
        x_arg = torch.relu(self.layer2(x))
        x = self.softmax(x_arg)
        logits = self.output_layer(x)
        return logits, x_arg
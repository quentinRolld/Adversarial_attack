import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

# Define a custom maxout layer
class Maxout(nn.Module):
    def __init__(self, in_channels, out_channels, pool_size):
        super(Maxout, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pool_size = pool_size
        self.linear = nn.Linear(in_channels, out_channels * pool_size)

    def forward(self, x):
        x = self.linear(x)
        x = x.view(-1, self.out_channels, self.pool_size)
        x, _ = torch.max(x, dim=2)
        return x

# Define a deep maxout network
class MaxoutNet(nn.Module):
    def __init__(self):
        super(MaxoutNet, self).__init__()
        self.maxout1 = Maxout(784, 256, 2) # input size is 28 x 28 for MNIST
        self.maxout2 = Maxout(256, 128, 2)
        self.maxout3 = Maxout(128, 64, 2)
        self.maxout4 = Maxout(64, 10, 2) # output size is 10 for MNIST

    def forward(self, x):
        x = x.view(-1, 784) # flatten the input
        x = self.maxout1(x)
        x = self.maxout2(x)
        x = self.maxout3(x)
        x = self.maxout4(x)
        return x


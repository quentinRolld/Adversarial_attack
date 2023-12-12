
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.autograd import Variable


# There we create the class for our Shallow RBF network


# Versoin 1:

class ShallowRBF(nn.Module):
    def __init__(self, input_dim, num_classes, num_centers):
        super(ShallowRBF, self).__init__()
        self.centers = nn.Parameter(torch.randn(num_centers, input_dim))
        self.beta = nn.Parameter(torch.ones(num_centers))
        self.fc = nn.Linear(num_centers, num_classes)

    def forward(self, x):
        
        x = x.view(x.size(0), -1)  # Flatten the input
        # Calculate the RBF activations
        rbf_activations = torch.exp(-self.beta * torch.norm(x.unsqueeze(1) - self.centers, dim=2))

        # Normalize the RBF activations
        rbf_activations = rbf_activations / torch.sum(rbf_activations, dim=1, keepdim=True)

        # Pass the normalized RBF activations through the linear layer
        output = self.fc(rbf_activations)

        return output
    

# Version 2:

class RbfNet2(nn.Module):
    def __init__(self, centers, num_class=10):
        super(RbfNet2, self).__init__()
        self.centers = centers
        self.num_centers = centers.size(0)
        self.num_class = num_class
    
        self.linear = torch.nn.Linear(self.num_centers, self.num_class, bias=True)
        self.beta = nn.Parameter(torch.ones(1,self.num_centers)/10)

    def radial_fun(self, batches):
        n_input = batches.size(0)
        x = self.centers.view(self.num_centers,-1).repeat(n_input,1,1)
        center = batches.view(n_input,-1).unsqueeze(1).repeat(1,self.num_centers,1)
        # Here we use the radial basis function, with the norm being the euclidian distance and 
        # the function being the gaussian function
        # h(x) = exp(-beta*||x-c||^2)
        hx = torch.exp(-self.beta.mul((x-center).pow(2).sum(2,keepdim=False).sqrt() ) )
        return hx
    
    def forward(self, batches):
        radial_val = self.radial_fun(batches)
        class_score = self.linear(radial_val)
        return class_score
    
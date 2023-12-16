import torch
import torch.nn as nn
import math
import numpy as np

"""
    On va d'abord implémenter une fonction d'activation Maxout : Maxout(x) = max(x1, x2, ..., xk)
    Class maxout récupéré sur le git https://github.com/thetechdude124/Custom-Maxout-Activation-Implementation/tree/master

"""

class CustomMaxout(nn.Module):

    #Initialize maxout function
    def __init__(self, j_inputs, k_output_groups, n_channels, bias = True):
        #Super initialization
        super(CustomMaxout, self).__init__()
        #Initialize instance variables
        self.j_inputs = j_inputs
        self.k_output_groups = k_output_groups
        self.n_channels = n_channels
        self.is_bias = bias

        #Generate tensor of channels * inputs and then output groups
        #We're creating tensors with the number of channels and output groups -> these are the FINAL WEIGHTS (output shape) of the activation function
        self.weights = torch.nn.Parameter(torch.Tensor(n_channels * k_output_groups, j_inputs))
        
        #If bias is true, create a new bias parameter -> this bias parameter is a VECTOR that will be added to each column-wise vector of the weights function if bias is enabled
        if bias: self.bias = torch.nn.Parameter(torch.Tensor(n_channels * k_output_groups))
        else: self.register_parameter('MAXOUT_BIAS', None)

        #Initilize weights
        self.initWeights()

    #Feedforward function
    def forward(self, x):
        #Apply linear transform (y = xA^T + b, output = x * transpose(weights) + bias)
        self.forward_tensor = torch.nn.functional.linear(x, self.weights, bias = self.bias if self.is_bias else None)
        #Now, take the MAXIUMUM of each row to yield a vector of dimensions (n_channels, j_inputs, k_outputs)
        #Use 1 dimensional max pooling to accomplish this
        #Set kernel size to two (window across where pooling will be applied)
        #Add an extra dimension at -3 -> maxpool accepts 3+ dimensions only. Do not unsqeueeze simply at one, as higher order tensors simply don't work (first dimension for 3+ degree tensors are not the 3rd dimension) 
        self.forward_tensor = torch.nn.functional.max_pool1d(torch.unsqueeze(self.forward_tensor, -3), kernel_size = self.n_channels)
        #Once pooling has been done, squeeze the tensor once again to get rid of the extra dimsension
        self.forward_tensor = torch.squeeze(self.forward_tensor)   
        return self.forward_tensor

    #For initializing weights on startup. This is INCREDIBLY CRITICAL - if we simply initialize weights randomly, exploding and diminishing gradients (perhaps more of the former) will occur frequently
    #We need to initialize weights in a uniform manner such that each time we initialize this function we can expect similar behaviour and similar overall performance
    def initWeights(self):
        #Best practice - initialize with a UNIFORM (not normal) distribution with values ranging +-1/sqrt(input_neurons)
        min = -1/math.sqrt(self.j_inputs)
        max = 1/math.sqrt(self.j_inputs)
        #Initialize uniform weights
        nn.init.uniform_(self.weights, min, max)
        #Initialize bias uniformly if present
        if (self.is_bias): nn.init.uniform_(self.bias, min, max)

#Test function by passing in a sample input
def sampleRun(mat_size, inputs, outputs, channels, bias):
    #Generate random matrix of specified size - convert to PyTorch tensor to allow for Pytorch linear transforms
    test_mat = np.random.rand(mat_size, mat_size)
    test_mat = torch.Tensor(test_mat)
    #Create new maxout_layer object with parameters
    maxout_layer = CustomMaxout(inputs, outputs, channels, bias = bias)
    #Call forward method and print output
    forward_tensor = maxout_layer.forward(test_mat)
    print(test_mat)
    print(forward_tensor)

#Test is only active when file is run
if __name__ == "__main__":
    sampleRun(mat_size = 10, inputs = 10, outputs = 5, channels = 1, bias = False)
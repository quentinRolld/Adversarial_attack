import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import fetch_openml
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from torchvision import datasets, transforms

""" Here is the function that will compute the adversarial examples """

def adv_attack(model, image, epsilon, labels,criterion):

    # Move the model to the CPU
    image = image.clone().detach().requires_grad_(True)  # Create a new tensor with requires_grad set to True
    labels = labels
    sorties = model(image)

    model.zero_grad()  # zero all the gradients
    cost = criterion(sorties, labels)
    cost.backward()  # compute the gradient of the loss

    # compute the sign of gradient
    sign_data_grad = image.grad.sign()
    # generate the altered image
    altered_image = image + epsilon * sign_data_grad

    # clipping to maintain [0, 1] range
    altered_image = torch.clamp(altered_image, 0, 1)

    return altered_image

def create_adv_test(model, X_test_tensor, y_test_tensor,eps,criterion):
    adv_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    adv_loader = DataLoader(adv_dataset, batch_size=len(adv_dataset), shuffle=False)
    for batch_X, batch_y in adv_loader:
        alt_im=adv_attack(model, batch_X, eps, batch_y,criterion)
    alt_im_norm =torch.tensor(alt_im, dtype=torch.float32)
    return alt_im_norm


""" class to add a random eps noise to the images"""

class AddUniformNoise(object):
    def __init__(self, epsilon):
        self.epsilon = epsilon
        
    def __call__(self, tensor):
        noise = torch.empty(tensor.size()).uniform_(-self.epsilon, self.epsilon)
        return tensor + noise
    
    def __repr__(self):
        return self.__class__.__name__ + '(epsilon={0})'.format(self.epsilon)

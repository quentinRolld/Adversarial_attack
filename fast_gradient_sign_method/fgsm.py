
# inspired from : https://pytorch.org/tutorials/beginner/fgsm_tutorial.html


import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import json
import torchvision.datasets as dsets
import torch.utils.data as Data


""" Here is the function that will compute the adversarial examples
arguments :
    - model : the model to test
    - image : the image to alter
    - epsilon : the value of epsilon for the adversarial attack
    - loss : the loss function to use
    - labels : the labels of the image """

def adv_attack(model, device, image, epsilon, labels):

    loss = nn.CrossEntropyLoss()

    # Move the model to the CPU
    image = image.to(device) # Move to CPU
    labels = labels.to(device) 
    image.requires_grad = True
    sorties = model(image)

    model.zero_grad() # zero all the gradients
    cost = loss(sorties, labels).to(device) # compute the loss
    cost.backward() # compute the gradient of the loss

    # compute the sign of gradient
    sign_data_grad = image.grad.sign()
    # generate the altered image
    altered_image = image + epsilon*sign_data_grad

    # clipping to maintain [0,1] range
    altered_image = torch.clamp(altered_image, 0, 1)

    return altered_image




""" This is the test function that will apply the adversarial attack 
on a model and return the accuracy, confidence and altered image
arguments :
    - model : the model to test
    - device : the device on which the model is
    - normal_loader : the loader of the normal images
    - epsilon : the value of epsilon for the adversarial attack """


def test( model, device, normal_loader, epsilon):

    # Accuracy counter
    correct = 0
    adv_examples = []

    # Loop over all examples in test set
    for image, labels in normal_loader:

        image, labels = image.to(device), labels.to(device)
        output = model(image)
        init_pred = output.max(1, keepdim=True)[1]

        # Loss calculation
        loss = F.nll_loss(output, labels)

        # Call FGSM Attack
        altered_data = adv_attack(model, device, image, epsilon, labels).to(device)

        # Re-classify the perturbed image
        output = model(altered_data)

        # Check for success
        final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        print("Taille de final_pred :", final_pred.size())
        print("Taille de labels :", labels.size())
        #print(final_pred.item())

        correct += final_pred.eq(labels.view_as(final_pred)).sum().item()

        """ if final_pred.item() == (labels.unsqueeze(1)).item():
            correct += 1
            # Special case for saving 0 epsilon examples
            if epsilon == 0 and len(adv_examples) < 5:
                adv_ex = altered_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )
        else:
            # Save some adv examples for visualization later
            if len(adv_examples) < 5:
                adv_ex = altered_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) ) """
        
        probabilities = nn.functional.softmax(output, dim=1)
        confidence, predictions = torch.max(probabilities, dim=1)

    # Calculate final accuracy for this epsilon
    final_acc = correct/float(len(normal_loader))
    print(f"Epsilon: {epsilon}\tTest Accuracy = {correct} / {len(normal_loader)} = {final_acc}")

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples, confidence
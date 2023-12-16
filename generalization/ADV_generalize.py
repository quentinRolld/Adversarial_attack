
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

import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from matplotlib.colors import LinearSegmentedColormap
from sklearn.preprocessing import LabelEncoder


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
    sorties, _ = model(image)

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
    - normal_loader : the loader of the normal images --> test loader
    - epsilon : the value of epsilon for the adversarial attack """


def test_Maxout( model_Maxout, device, normal_loader, epsilon):

    # Accuracy counter
    correct = 0
    adv_examples = []

    # Loop over all examples in test set
    for image, labels in normal_loader:

        image, labels = image.to(device), labels.to(device)
        output, _ = model_Maxout(image)
        

        # Call FGSM Attack
        altered_data = adv_attack(model_Maxout, device, image, epsilon, labels).to(device)

        # Re-classify the perturbed image using the new images wihch are now altered
        output, output_arg_softmax = model_Maxout(altered_data)
        print("arg softmax :", output_arg_softmax)
        print("output :", output.shape)

        # Check for success
        final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability --> argmax
        _, predicted = torch.max(output, 1)
        
        correct = (final_pred == labels).sum().item()

        probabilities = nn.functional.softmax(output, dim=1)
        confidence, predictions = torch.max(probabilities, dim=1) # le max de la sortie de la fonction softmax
                                                                  # pour trouver la confiance et la prediction
                                                                
    # Calculate final accuracy for this epsilon
    final_acc = correct/float(len(normal_loader))
    print(f"Epsilon: {epsilon}\tTest Accuracy = {correct} / {len(normal_loader)} = {final_acc}")

    # Return the accuracy and the arg of the softmax
    return final_acc, output



def training_loop_Maxout(optimizer, model, criterion, X_train_tensor, y_train_tensor, num_epochs=200, batch_size=128):
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)

    train_losses = []  # Store training losses for each epoch

    for epoch in range(num_epochs):
        for batch_X, batch_y in train_loader:
            outputs, _ = model(batch_X)
            loss = criterion(outputs, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_losses.append(loss.item())  # Store the loss after each epoch
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    return train_losses,model

def eval_test_Maxout(X_test_tensor,y_test_tensor,model):
    # Convert test data to DataLoader for batching
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model.eval()  # Set the model to evaluation mode
    total = 0
    correct = 0
    total_mean_confidence = 0.0
    total_samples = 0

    incorrect_mean_confidence = 0.0
    incorrect_samples = 0

    args_x = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs, arg_x = model(batch_X)
            args_x.append(arg_x)
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

            # Calculate mean confidence for all predictions
            probabilities = nn.functional.softmax(outputs, dim=1)
            confidences, _ = torch.max(probabilities, dim=1)
            total_mean_confidence += confidences.sum().item()
            total_samples += batch_y.size(0)

            # Calculate mean confidence for incorrect predictions
            incorrect_mask = predicted != batch_y
            if incorrect_mask.sum().item() > 0:
                incorrect_mean_confidence += confidences[incorrect_mask].sum().item()
                incorrect_samples += incorrect_mask.sum().item()

    # Calculate mean confidence for all examples
    if total_samples > 0:
        total_mean_confidence /= total_samples

    # Calculate mean confidence for incorrect predictions
    if incorrect_samples > 0:
        incorrect_mean_confidence /= incorrect_samples

    accuracy = correct / total
    print(f'Accuracy: {accuracy * 100:.2f}%')
    print(f'Mean Confidence for All Examples: {total_mean_confidence:.4f}')
    print(f'Mean Confidence for Incorrect Predictions: {incorrect_mean_confidence:.4f}')

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
import torch.nn.functional as F
from maxout import CustomMaxout
from adv_attack import create_adv_test, AddUniformNoise
import matplotlib.pyplot as plt
import numpy as np

def transform_tensor(noise=False):
    """applique les transformations sur les images

    Args:
        x (tensor): images
        noise (bool, optional): ajoute un bruit epsilon de 0.1 selon une loie uniforme. Defaults to False.
    """
    if noise:
        transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)) # Normalize the images to [-1, 1]
        ])
    else:
        transform = transforms.Compose([
        transforms.ToTensor(),
        AddUniformNoise(0.1), #on utilise la class pour ajouter un bruit epsilon de 0.1 selon une loie uniforme
        transforms.Normalize((0.5,), (0.5,)) # Normalize the images to [-1, 1]
        ])
    return transform

def define_model(model=240):
    """définit le modèle

    Args:
        model (int, optional): nombre d'unité par couche. Defaults to 240.
    """
    n_channels = 1
    dropout = 0.5   
    if model == 240: 
        #on créé le premier model qui à 240 unit per layer model
        Maxout_240U_Model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28*28*n_channels, 240),
        CustomMaxout(240, 200, n_channels, True),
        nn.Dropout(dropout),
        nn.Linear(200, 160),
        CustomMaxout(160, 120, n_channels, True),
        nn.Dropout(dropout),
        nn.Linear(120, 80),
        CustomMaxout(80, 40, n_channels, True),
        nn.Dropout(dropout),
        nn.Linear(40, 10),
        nn.LogSoftmax(dim=1)
        )
        return Maxout_240U_Model
    
    elif model == 1600:
        Maxout_1600U_Model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28*28*n_channels, 1600),
        CustomMaxout(1600, 1500, n_channels, True),
        nn.Dropout(dropout),
        nn.Linear(1500, 1400),
        CustomMaxout(1400, 1300, n_channels, True),
        nn.Linear(1300, 1200),
        CustomMaxout(1200, 1100, n_channels, True),
        nn.Dropout(dropout),
        nn.Linear(1100, 1000),
        CustomMaxout(1000, 800, n_channels, True),
        nn.Dropout(dropout),
        nn.Linear(800, 600),
        CustomMaxout(600, 400, n_channels, True),
        nn.Dropout(dropout),
        nn.Linear(400, 200),  
        CustomMaxout(200, 100, n_channels, True),
        nn.Dropout(dropout),
        nn.Linear(100, 50),
        CustomMaxout(50, 25, n_channels, True),  
        nn.Dropout(dropout),
        nn.Linear(25, 10),  
        nn.LogSoftmax(dim=1)
        )
        return Maxout_1600U_Model
    
# Fonction de perte standard
def loss_fn(model, x, y):
    output = model(x)
    return F.cross_entropy(output, y)

# Fonction de perte adversariale
def adversarial_loss_fn(model, x, y, epsilon, alpha):
    """
    défini la fonction de perte adversariale
    Arguments:
        model (torch.nn.Module): modèle à entraîner
        x (torch.Tensor): données d'entrée
        y (torch.Tensor): étiquettes cibles
        epsilon (float): taille de l'attaque
        alpha (float): poids de la perte standard
    """
    # Calcul de la perte standard
    standard_loss = loss_fn(model, x, y)
    
    # Génération de l'exemple adverse
    x_adv = x + epsilon * torch.sign(torch.autograd.grad(standard_loss, x, create_graph=True)[0])
    
    # Calcul de la perte sur l'exemple adverse
    adversarial_loss = loss_fn(model, x_adv, y)
    
    # Combinaison des deux pertes
    return alpha * standard_loss + (1 - alpha) * adversarial_loss

def train_loop(dataloader, model, epsilon, alpha, adv_loss=True,):
    # Initialize the early stopping variables
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epsilon = 0.1
    alpha = 0.5
    # Define the number of epochs
    n_epochs = 60


    train_dataloader = dataloader

    #Train the model
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_losses = []
    valid_losses = []
    train_accuracies = []
    valid_accuracies = []
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0
        correct_train_preds = 0
        total_train_preds = 0
        for batch in train_dataloader:
            inputs, labels = batch
            # Move the inputs and labels to the device
            inputs = inputs.to(device).requires_grad_()
            labels = labels.to(device).requires_grad_()

            optimizer.zero_grad()
            outputs = model(inputs)
            if adv_loss:
                loss = adversarial_loss_fn(model, inputs, labels, epsilon, alpha)
            else:
                loss = loss_fn(model, inputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total_train_preds += labels.size(0)
            correct_train_preds += (predicted == labels).sum().item()

        train_losses.append(train_loss / len(train_dataloader))
        train_accuracies.append(100 * correct_train_preds / total_train_preds)
        
    # Plot the training losses and accuracies    
    fig, ax1 = plt.subplots(figsize=(12, 4))

    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    if adv_loss:
        ax1.set_ylabel('Training loss with adversarial loss', color=color)
    else:
        ax1.set_ylabel('Training loss with crossentropy_loss', color=color)
    ax1.plot(train_losses, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('Training accuracy', color=color) 
    ax2.plot(train_accuracies, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    plt.show()

    return model

def test_loop(model, test_dataloader, adv_loss=True):
    epsilon, alpha = 0.1, 0.5
    total_test_preds = 0
    correct_test_preds = 0
    test_loss = 0
    misclassified_confidences = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    model.eval()  
    for batch in test_dataloader:
        
        inputs, labels = batch
        inputs = inputs.to(device).requires_grad_()
        labels = labels.to(device)
    
        # Calculate adversarial loss and generate adversarial examples
        if adv_loss:    
            loss = adversarial_loss_fn(model, inputs, labels, epsilon, alpha)
        else:
            loss = loss_fn(model, inputs, labels)
        test_loss += loss.item()
    
        # Disable gradient calculation for prediction and accuracy calculation
        with torch.no_grad():
            outputs = model(inputs)
            probabilities = torch.exp(outputs)  # Convert log probabilities to probabilities
            _, predicted = torch.max(probabilities.data, 1)
    
        total_test_preds += labels.size(0)
        correct_test_preds += (predicted == labels).sum().item()
    
        # Calculate confidence
        confidence = torch.max(probabilities, dim=1)[0]
    
        # Store confidence of misclassified examples
        misclassified = predicted != labels
        misclassified_confidences.extend(confidence[misclassified].tolist())
    
    test_accuracy = 100 * correct_test_preds / total_test_preds
    print(f'Test loss: {test_loss / len(test_dataloader):.3f}.. '
        f'Test accuracy: {test_accuracy:.3f}')
    print(f'Average confidence of misclassified examples: {np.mean(misclassified_confidences):.3f}\n')
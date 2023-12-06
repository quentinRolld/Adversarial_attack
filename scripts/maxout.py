import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

"""
    On va d'abord implémenter une fonction d'activation Maxout : Maxout(x) = max(x1, x2, ..., xk)
    on défini un CNN avec des fonctions d'activation maxout pour

"""

# Définition de la fonction d'activation Maxout
class Maxout(nn.Module):
    def __init__(self, num_units):
        super(Maxout, self).__init__()
        self.num_units = num_units

    def forward(self, x):
        # x.shape = (batch_size, channels, height, width)
        # Reshape input to (batch_size, num_units, -1) and take max over last dimension
        x = x.view(*x.shape[:2], self.num_units, -1)
        x, _ = x.max(-1)
        return x

# Définition du CNN avec des fonctions d'activation maxout 
class MaxoutCNN(nn.Module):
    def __init__(self, dropout_rate=0.4):
        super(MaxoutCNN, self).__init__()
        self.architecture = nn.Sequential(
            nn.Conv2d(1, 6 * 2, 5, padding=2), # Multiplier par 2 pour Maxout
            Maxout(num_units=6),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout_rate), # Dropout pour les couches convolutives
            nn.Conv2d(6, 16 * 2, 5), # Multiplier par 2 pour Maxout
            Maxout(num_units=16),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout_rate), # Dropout pour les couches convolutives
            nn.Flatten(),
            nn.Linear(16 * 5 * 5 * 2, 120 * 2), # Multiplier par 2 pour Maxout
            Maxout(num_units=120),
            nn.Dropout(dropout_rate), # Dropout pour les couches linéaires
            nn.Linear(120, 84 * 2), # Multiplier par 2 pour Maxout
            Maxout(num_units=84),
            nn.Dropout(dropout_rate) # Dropout avant la couche de classification
        )

        self.classifier = nn.Linear(84, 10) # Couche de classification

    def forward(self, x):
        x = self.architecture(x)
        return self.classifier(x)
    
CNN = MaxoutCNN()
print(CNN)
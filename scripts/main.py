import methodes
from torch.utils.data import DataLoader
from torchvision import datasets
import torch.nn as nn
import torch
from adv_attack import create_adv_test

#on créé une transform normale et une transoform avec bruit
transform = methodes.transform_tensor(noise=False)
transform_noise = methodes.transform_tensor(noise=True)


training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=transform
)

test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=transform
)

training_data_noised = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=transform_noise
)

batch_size = 64
training_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
training_dataloader_noised = DataLoader(training_data_noised, batch_size=batch_size, shuffle=True)

#je définis le deep neural network maxout avec 240 unités par couche
DNN_240 = methodes.define_model(model=240)

#on réalise dans un premier temps un entrainement normal sans attaque
DNN_240_normal_training=methodes.train_loop(training_dataloader, DNN_240, epsilon=0.1, alpha=0.5, adv_loss=False,)

#on réalise ensuite un entrainement avec attaque
DNN_240_adv_training=methodes.train_loop(training_dataloader, DNN_240, epsilon=0.1, alpha=0.5, adv_loss=True,)

#enfin on réalise un entrainement sur une base d'entraînement bruitée
DNN_240_noised_training=methodes.train_loop(training_dataloader_noised, DNN_240, epsilon=0.1, alpha=0.5, adv_loss=True)

#on va maintenant créer la base de test adversariale

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#on récupère les images et les labels de la base de test
x_test = torch.cat([images for images, labels in test_dataloader]).to(device) #images de la base test
y_test = torch.cat([labels for images, labels in test_dataloader]).to(device) #labels de la base test

eps = 0.1
loss_func = nn.CrossEntropyLoss()

#création des images adverses pour le réseau  à 240 unites et une normale loss
altered_test_240U = create_adv_test(DNN_240_normal_training, x_test, y_test, eps, loss_func)

#création des images adverses pour le réseau  à 240 unites et une loss adversariale
altered_test_240U_adv = create_adv_test(DNN_240_adv_training, x_test, y_test, eps, loss_func, adv_loss=True)

#création des images adverses pour le réseau  à 240 unites et une loss adversariale sur une base d'entrainement bruitée
altered_test_240U_noised = create_adv_test(DNN_240_noised_training, x_test, y_test, eps, loss_func, adv_loss=True)



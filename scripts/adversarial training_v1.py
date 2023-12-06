# Encodage d'une fonction résistante à l'attaque adversariale pour un réseau profond
# On entraîne un maxout network régularisé avec un dropout
# Entraîner une fonction de coût basé sur la FGSM permet de régulariser correctement
# Update en continue l'apport en exemple adversariaux

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from maxout import MaxoutNet

# On défini la fonction d'attaque FGSM
def fgsm_attack(x, epsilon, data_grad):
    """
    x: input
    epsilon: perturbation magnitude
    data_grad: gradient of the loss w.r.t the input
    """
    return torch.clamp(x + epsilon * data_grad.sign(), 0, 1)

# on défini la fonction de coût
def cost_function(alpha, output, target, perturbed_output):
    """
    alpha: taux 
    output: output of the network on the original inputs
    target: target labels
    perturbed_output: output of the network on the perturbed inputs
    """
    loss = nn.CrossEntropyLoss()
    original_loss = loss(output, target)
    adversarial_loss = loss(perturbed_output, target)
    return alpha * original_loss + (1 - alpha) * adversarial_loss

# Load the MNIST dataset
transform = transforms.ToTensor()
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

# Create an instance of the network
net = MaxoutNet()

# Define the optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

# Train the network with FGSM attack and the cost function
num_epochs = 10
epsilon = 0.1 # the perturbation magnitude
alpha = 0.5 # the trade-off hyperparameter
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs.requires_grad = True  # Enable computation of gradients for inputs
        optimizer.zero_grad()
        outputs = net(inputs)
        # compute the gradient of the original loss w.r.t the input
        outputs.backward(torch.ones(outputs.size()), retain_graph=True)
        data_grad = inputs.grad.data
        # generate the perturbed inputs
        perturbed_inputs = fgsm_attack(inputs, epsilon, data_grad)
        # compute the output on the perturbed inputs
        perturbed_outputs = net(perturbed_inputs)
        # compute the cost function
        loss = cost_function(alpha, outputs, labels, perturbed_outputs)
        # update the model parameters
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 200 == 199: # print every 200 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 200))
            running_loss = 0.0

# Test the network on the original and perturbed test data
correct = 0
total = 0
adv_correct = 0
adv_total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        # compute the gradient of the original loss w.r.t the input
        outputs.backward(torch.ones(outputs.size()), retain_graph=True)
        data_grad = images.grad.data
        # generate the perturbed inputs
        perturbed_images = fgsm_attack(images, epsilon, data_grad)
        # compute the output on the perturbed inputs
        perturbed_outputs = net(perturbed_images)
        _, adv_predicted = torch.max(perturbed_outputs, 1)
        adv_total += labels.size(0)
        adv_correct += (adv_predicted == labels).sum().item()

print('Accuracy of the network on the 10000 original test images: %d %%' % (100 * correct / total))
print('Accuracy of the network on the 10000 perturbed test images: %d %%' % (100 * adv_correct / adv_total))

        
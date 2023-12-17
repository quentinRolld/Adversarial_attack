import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from matplotlib.colors import LinearSegmentedColormap
from sklearn.preprocessing import LabelEncoder

def prep_data(data):

    # Assuming y is your target labels
    y = data['target'].values

    # Normalize the data
    data_normalized = data.iloc[:, :-1].values / 255.0

    # Convert data to PyTorch tensors
    X_tensor = torch.tensor(data_normalized, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    # Split the data into training and testing sets
    X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)
    num_classes = len(set(y))
    print("Unique classes in target labels:", num_classes)


    return X_train_tensor,y_train_tensor,X_test_tensor,y_test_tensor,num_classes



def create_model(MaxoutNetworkWithSoftmax):
    input_size = 784
    num_classes = 10
    device = torch.device("cpu")

    # Define the model
    model = MaxoutNetworkWithSoftmax(input_size, num_classes)

    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Define the loss function
    criterion = nn.CrossEntropyLoss()
    return model,criterion,optimizer

def create_model_2(DeepMaxoutNetwork,X_train_tensor):

        # Hyperparameters
    num_epochs = 100
    Batch_size =  100
    # Create an instance of the DeepMaxoutNetwork
    deep_maxout_model = DeepMaxoutNetwork(input_dim=784, hidden_dim=100, output_dim=10, num_units=2, num_layers=3)
    input_size = X_train_tensor.shape[1]
    criterion = nn.CrossEntropyLoss()
    sgd =  optim.SGD(deep_maxout_model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    adam = optim.Adam(deep_maxout_model.parameters(), lr=0.01,weight_decay=1e-2) 
    optimizer = adam

    return deep_maxout_model,criterion,optimizer, num_epochs, Batch_size
 
def create_model_3(ShallowRBF,X_train_tensor,centers):
    # Hyperparameters
    num_epochs = 100
    learning_rate = 0.01
    momentum = 0.9
    Batch_size =  100
    num_classes = 10
    input_size = 784
    n_channels = 3
    num_units = 2
    # Create an instance of the DeepMaxoutNetwork
    RBF_model = ShallowRBF(input_dim=784, num_classes=10, num_centers=centers.shape[0])
    input_size = X_train_tensor.shape[1]
    criterion = nn.CrossEntropyLoss()
    sgd =  optim.SGD(RBF_model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    adam = optim.Adam(RBF_model.parameters(), lr=0.01) 
    optimizer = adam

    return RBF_model,criterion,optimizer, num_epochs, Batch_size

def create_model_4(ShallowSoftmaxNetBN,X_train_tensor):
    # Hyperparameters
    num_epochs = 100
    learning_rate = 0.01
    momentum = 0.9
    Batch_size = 100
    num_classes = 10
    input_size = 784
    n_channels = 3
    num_units = 2
    # Create an instance of the DeepMaxoutNetwork
    ShallowSoftmax_model = ShallowSoftmaxNetBN(input_dim=784, output_dim=10)

    input_size = X_train_tensor.shape[1]
    criterion = nn.CrossEntropyLoss()
    sgd =  optim.SGD(ShallowSoftmax_model.parameters(), lr=0.01,weight_decay=1e-2)
    adam = optim.Adam(ShallowSoftmax_model.parameters(), lr=0.01,weight_decay=1e-4)
    optimizer = adam
    return ShallowSoftmax_model,criterion,optimizer, num_epochs, Batch_size




def test_eps(test_Maxout,model,device,test_dataloader):
    # Run test for each epsilon

    accuracies = []
    args = []
    epsilons = [-15,-14,-13,-12,-11, -10,-9,-8,-7,-6, -5,-4,-3,-2,-1, 0,1,2,3,4, 5,6,7,8,9, 10,11,12,13,14, 15]

    for eps in epsilons:
        accuracy, arg = test_Maxout(model, device, test_dataloader, eps)
        accuracies.append(accuracy)
        args.append(arg)
    return args, epsilons

def plot_eps(args,epsilons):
    for i in range(len(args)):
        args[i] = args[i].detach().numpy()
        args[i] = np.log(args[i]) - np.log(np.sum(np.exp(args[i]), axis=1, keepdims=True))
        args[i] = args[i].mean(axis=0)
    # Plot the average values as a function of epsilon
    #plt.figure(figsize=(6, 18))  # Increase the height by a factor of 3
    plt.plot(epsilons, [i[:] for i in args])
    plt.xlabel('Epsilon')
    plt.ylabel('softmax output')

    plt.title('softmax output for each class vs Epsilon')
    plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
    plt.show()



def training_loop(optimizer, model, criterion, X_train_tensor, y_train_tensor, num_epochs=200, batch_size=128):
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)

    train_losses = []  # Store training losses for each epoch

    for epoch in range(num_epochs):
        for batch_X, batch_y in train_loader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_losses.append(loss.item())  # Store the loss after each epoch
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    return train_losses,model


def plot_losses(train_losses):
    import matplotlib.pyplot as plt
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Curve')
    plt.legend()
    plt.show()

def eval_train(X_train_tensor,y_train_tensor,model):

    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        train_outputs = model(X_train_tensor)
        _, predicted = torch.max(train_outputs, 1)
        correct = (predicted == y_train_tensor).sum().item()
        total = y_train_tensor.size(0)
        train_accuracy = correct / total

    print(f'Training Accuracy: {train_accuracy * 100:.2f}%')


def visualize_weights_and_signs(model):
    # Get the weights from the first layer (assuming it's the layer you're interested in)
    weights = model.linear.weight.data

    # Extract the signs of the weights
    weight_signs = torch.sign(weights)

    # Check if the weights are 1-dimensional
    if weights.dim() == 1:
        # Reshape the weights to be a 2D tensor with one row
        weights = weights.view(1, -1)

    # Reshape the weights to match the original image dimensions (assuming 28x28)
    weight_images = weights.view(-1, 28, 28)

    # Reshape the weight signs to match the original image dimensions (assuming 28x28)
    weight_sign_images = weight_signs.view(-1, 28, 28)

    # Plot each set of weights and weight signs in a separate subplot
    num_classes = weight_images.size(0)
    fig, axes = plt.subplots(num_classes, 2, figsize=(16, 8 * num_classes))

    for i in range(num_classes):
        # Plot weights
        axes[i, 0].imshow(weight_images[i].cpu().numpy(), cmap='gray')
        axes[i, 0].set_title(f'Class {i} - Weight')
        axes[i, 0].axis('off')

        # Plot weight signs
        axes[i, 1].imshow(weight_sign_images[i].cpu().numpy(), cmap='gray', vmin=-1, vmax=1)
        axes[i, 1].set_title(f'Class {i} - Sign')
        axes[i, 1].axis('off')

    # Show the plot
    plt.show()

def eval_test(X_test_tensor, y_test_tensor, model):
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

    wrong_predictions = []
    correct_predictions = []
    all_predictions = []  # List to store all predictions

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

            # Store all predictions
            all_predictions.extend(predicted.tolist())

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

                # Store the wrong predictions
                wrong_predictions.extend(predicted[incorrect_mask].tolist())

            # Store the correct predictions
            correct_predictions.extend(predicted[~incorrect_mask].tolist())

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

    return wrong_predictions, correct_predictions, all_predictions  # Return all predictions
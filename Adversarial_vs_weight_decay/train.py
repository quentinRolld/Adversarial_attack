import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_openml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from matplotlib.colors import LinearSegmentedColormap
from sklearn.preprocessing import LabelEncoder


def prep_data(filtered_data):

    X = filtered_data.iloc[:, :-1].values
    y = filtered_data['target'].values

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Assuming y_train is your target labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)

    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_encoded, dtype=torch.long)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    print("Unique classes in target labels:", set(y_train))
    return X_train,y_train,X_train_tensor,y_train_tensor,X_test_tensor,y_test_tensor


def create_model(X_train,y_train):
    class LogisticRegressionModel(nn.Module):
        def __init__(self, input_size, num_classes):
            super(LogisticRegressionModel, self).__init__()
            self.linear = nn.Linear(input_size, num_classes)

        def forward(self, x):
            return self.linear(x)

    # Specify input size and dynamically set the number of classes
    input_size = X_train.shape[1]
    num_classes = len(set(y_train))

    # Instantiate the model
    model = LogisticRegressionModel(input_size, num_classes)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    return model,criterion,optimizer

def training_loop(optimizer, model, criterion, X_train_tensor, y_train_tensor,num_epochs=200,batch_size=128):

    # Convert data to DataLoader for batching
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)

    # Training loop
    for epoch in range(num_epochs):
        for batch_X, batch_y in train_loader:
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Print the loss after each epoch
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
    return model


def map_values(predicted_tensor):
    # Define your mapping here
    mapping = {0: 3, 1: 7}

    # Apply the mapping to the predicted tensor
    mapped_tensor = torch.tensor([mapping[val.item()] for val in predicted_tensor])

    return mapped_tensor

def eval_test(X_test_tensor,y_test_tensor,model):
    # Convert test data to DataLoader for batching
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Evaluation loop
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)
            _, predicted = torch.max(outputs.data, 1)
            predicted = map_values(predicted)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

    accuracy = correct / total
    print(f'Test Accuracy: {accuracy * 100:.2f}%')

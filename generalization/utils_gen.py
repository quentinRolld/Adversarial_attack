
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_openml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from matplotlib.colors import LinearSegmentedColormap
import numpy as np


def extract_from_mnist():

   # Load the MNIST dataset
    mnist = fetch_openml('mnist_784')

    # Convert labels to integers
    mnist.target = mnist.target.astype(int)

    data = pd.DataFrame(data=mnist.data, columns=mnist.feature_names)
    data['target'] = mnist.target

    data = data.sample(n=20000, random_state=42)

    # Reset the index of the filtered dataset
    data = data.reset_index(drop=True)

    # Print the shape of the filtered dataset
    print("Filtered dataset shape:", data.shape)
    print("Labels:", data['target'])

    return data



def visualize_data(num_rows,num_cols,filtered_data):

    num_rows = 10
    num_cols = 10
    cmap = LinearSegmentedColormap.from_list('custom_gray', [(0.5, 0.5, 0.5), (1, 1, 1)], N=256)

    # Create a figure with subplots and set the face color to gray
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(15, 15))
    fig.set_facecolor('gray')

    # Iterate over the samples in the filtered dataset
    for i, ax in enumerate(axes.flat):
        # Calculate the row and column indices
        row_index = i // num_cols
        col_index = i % num_cols
        

        if i < len(filtered_data):

            if isinstance(filtered_data, pd.DataFrame):
                # Use iloc for DataFrame
                image = filtered_data.iloc[i, :-1].values.reshape(28, 28)
            elif isinstance(filtered_data, np.ndarray):
                # Use [i] for NumPy array
                image = filtered_data[i].reshape(28, 28)
            else:
                print("Unsupported data type")
            # Plot the image with the custom gray background
            ax.imshow(image, cmap=cmap, vmin=0, vmax=1)
            ax.axis('off')
        
        

    # Adjust the spacing between subplots
    plt.tight_layout()

    # Show the plot
    plt.show()
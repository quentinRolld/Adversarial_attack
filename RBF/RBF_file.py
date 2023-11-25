# Importations
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import fetch_openml

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the functions from fgsm.py
from .fgsm_copie import adv_attack, test
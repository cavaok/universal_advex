import torch
from torch import optim
from torch import nn
import torch.nn.functional as F
import os
from helper import create_diffuse_one_hot
from data import get_mnist_loaders

# Get data for training and testing
train_loader, test_loader, _ = get_mnist_loaders()

# Constants
image_dim = 28 * 28
num_classes = 10
input_dim = image_dim + num_classes

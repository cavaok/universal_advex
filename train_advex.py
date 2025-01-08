import torch
from torch import optim
from torch import nn
import torch.nn.functional as F
from helper import create_diffuse_one_hot, set_DoC
from data import get_mnist_digit_loaders
from auto import load_autoencoder64_model, load_autoencoder128_model, load_autoencoder256_model, load_autoencoder512_model, load_funky_autoencoder_model
from mlp import load_mlp_model
from hadamard import load_hadamard_model


# Function for training adversarial examples (for autoencoders & hadamards)
#     takes in data loaders and the model
#     returns metrics to log about the training process


# Function for training mlp adversarial examples
#     takes in data loaders and the model
#     returns metrics to log about the training process

# Function for logging metrics

# Main function
#     load all desired pre-trained models
#     set all pre-trained models to eval mode
#     create a list of pre-trained models (not including MLP)
#     then calls the functions for training adversarial examples
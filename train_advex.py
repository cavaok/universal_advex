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

def load_all_models():
    models = {}

    # Load MLP model
    models['mlp'] = load_mlp_model("models_backup/mlp_elu_512_256_128.pth")

    # Load base autoencoder64 model (single iteration version)
    models['auto64_1'] = load_autoencoder64_model(
        "models_backup/encoder_auto_64_elu_1.pth",
        "models_backup/decoder_auto_64_elu_1.pth"
    )

    # Load autoencoder64 sum models (2-4 iterations)
    for x in range(2, 5):
        models[f'auto64_{x}'] = load_autoencoder64_model(
            f"models_backup/encoder_auto_64_elu_{x}_sum.pth",
            f"models_backup/decoder_auto_64_elu_{x}_sum.pth"
        )

    # Load base autoencoder128 model (single iteration version)
    models['auto128_1'] = load_autoencoder128_model(
        "models_backup/encoder_auto_128_elu_1.pth",
        "models_backup/decoder_auto_128_elu_1.pth"
    )

    # Load autoencoder128 sum models (2-7 iterations)
    for x in range(2, 8):
        models[f'auto128_{x}'] = load_autoencoder128_model(
            f"models_backup/encoder_auto_128_elu_{x}_sum.pth",
            f"models_backup/decoder_auto_128_elu_{x}_sum.pth"
        )

    # Load base autoencoder256 model (single iteration version)
    models['auto256_1'] = load_autoencoder256_model(
        "models_backup/encoder_auto_256_elu_1.pth",
        "models_backup/decoder_auto_256_elu_1.pth"
    )

    # Load autoencoder256 sum models (2-10 iterations)
    for x in range(2, 11):
        models[f'auto256_{x}'] = load_autoencoder256_model(
            f"models_backup/encoder_auto_256_elu_{x}_sum.pth",
            f"models_backup/decoder_auto_256_elu_{x}_sum.pth"
        )

    # Load base autoencoder512 model (single iteration version)
    models['auto512_1'] = load_autoencoder512_model(
        "models_backup/encoder_auto_512_elu_1.pth",
        "models_backup/decoder_auto_512_elu_1.pth"
    )

    # Load autoencoder512 sum models (2-10 iterations)
    for x in range(2, 11):
        models[f'auto512_{x}'] = load_autoencoder512_model(
            f"models_backup/encoder_auto_512_elu_{x}_sum.pth",
            f"models_backup/decoder_auto_512_elu_{x}_sum.pth"
        )

    # Load funky autoencoder models (1-10 iterations)
    for x in range(1, 11):
        models[f'funkyauto_{x}'] = load_funky_autoencoder_model(
            f"models_backup/model_794_elu_{x}.pth"
        )

    # Load hadamard models (1-3 iterations)
    for x in range(1, 4):
        models[f'hadamard_{x}'] = load_hadamard_model(
            f"models_backup/f1_hadamard_{x}.pth",
            f"models_backup/f2_hadamard_{x}.pth"
        )

    return models


if __name__ == "__main__":
    # Test loading all models
    models = load_all_models()
    print("Successfully loaded models:")
    for model_name in models.keys():
        print(f"- {model_name}")
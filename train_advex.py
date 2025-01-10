import torch
import pickle
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

""" without print statements
def load_adversarial_cases():
    # Load pickle data
    with open('doc.pkl', 'rb') as f:
        data = pickle.load(f)

    # Get digit loaders
    digit_loaders = get_mnist_digit_loaders(batch_size=1)

    cases_per_image = 18
    images_per_digit = 5
    all_cases = []

    for digit in range(10):
        digit_iterator = iter(digit_loaders[digit])

        for image_idx in range(images_per_digit):
            image, label = next(digit_iterator)

            start_idx = (digit * images_per_digit + image_idx) * cases_per_image
            end_idx = start_idx + cases_per_image
            current_cases = data[start_idx:end_idx]

            for case in current_cases:
                all_cases.append((image, label, case[0], case[1]))

    return all_cases
"""

def load_adversarial_cases():
    """
    Loads and organizes all adversarial cases with their corresponding MNIST images.
    Returns:
        list of tuples: [(image, label, digit, target_distribution), ...]
    """
    print("\nStarting to load adversarial cases...")

    # Load pickle data
    with open('doc.pkl', 'rb') as f:
        data = pickle.load(f)
    print(f"Loaded pickle file with {len(data)} total cases")

    # Get digit loaders
    digit_loaders = get_mnist_digit_loaders(batch_size=1)
    print("Created MNIST digit loaders")

    cases_per_image = 18
    images_per_digit = 5
    all_cases = []

    for digit in range(10):
        print(f"\nProcessing digit {digit}:")
        digit_iterator = iter(digit_loaders[digit])

        for image_idx in range(images_per_digit):
            image, label = next(digit_iterator)
            print(f"  Image {image_idx + 1}/5 for digit {digit}:")
            print(f"    - Image shape: {image.shape}")
            print(f"    - Label from loader: {label}")

            start_idx = (digit * images_per_digit + image_idx) * cases_per_image
            end_idx = start_idx + cases_per_image
            current_cases = data[start_idx:end_idx]

            # Print first target distribution to verify
            print(f"    - First target distribution shape: {current_cases[0][1].shape}")
            print(f"    - Processing {len(current_cases)} cases for this image")

            for case in current_cases:
                # Verify digit matches label
                if case[0] != label:
                    print(f"    WARNING: Pickle digit {case[0]} doesn't match loader label {label}")
                all_cases.append((image, label, case[0], case[1]))

    print(f"\nFinished loading all cases. Total cases: {len(all_cases)}")

    # Verify final count
    expected_count = 10 * 5 * 18  # 10 digits * 5 images * 18 cases
    if len(all_cases) != expected_count:
        print(f"WARNING: Expected {expected_count} cases but got {len(all_cases)}")
    else:
        print("Case count matches expected total of 900")

    return all_cases
if __name__ == "__main__":
    # Load in all models and save to one dictionary
    models = load_all_models()
    for model_name in models.keys(): # doublechecking they saved successfully
        print(f"- {model_name}")

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Move all models to device and freeze params
    for name, model in models.items():
        model.to(device)
        for param in model.parameters():
            param.requires_grad = False
        print(f"Moved {name} to {device} and froze parameters")

    # Set up adversarial


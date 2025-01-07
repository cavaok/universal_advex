import torch
from torch import nn, optim
import torch.nn.functional as F
import pandas as pd
from datetime import datetime
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('/home/okcava/projects/universal_advex')
from helper import create_diffuse_one_hot
from data import get_mnist_loaders

# Configuration for all model variants
MODEL_CONFIGS = {
    '512_elu': {
        'batch_size': 128,
        'learning_rate': 0.00095101,
        'lambda': 0.43752,
        'sizes': [512],
        'activation': nn.ELU
    },
    '512_sigmoid': {
        'batch_size': 32,
        'learning_rate': 0.0028379,
        'lambda': 0.3263,
        'sizes': [512],
        'activation': nn.Sigmoid
    },
    '256_elu': {
        'batch_size': 128,
        'learning_rate': 0.00052486,
        'lambda': 0.19035,
        'sizes': [512, 256],
        'activation': nn.ELU
    },
    '256_sigmoid': {
        'batch_size': 32,
        'learning_rate': 0.0010224,
        'lambda': 0.069437,
        'sizes': [512, 256],
        'activation': nn.Sigmoid
    },
    '128_elu': {
        'batch_size': 64,
        'learning_rate': 0.00085713,
        'lambda': 0.23551,
        'sizes': [512, 256, 128],
        'activation': nn.ELU
    },
    '128_sigmoid': {
        'batch_size': 32,
        'learning_rate': 0.00092193,
        'lambda': 0.082284,
        'sizes': [512, 256, 128],
        'activation': nn.Sigmoid
    },
    '64_elu': {
        'batch_size': 64,
        'learning_rate': 0.00098095,
        'lambda': 0.45371,
        'sizes': [512, 256, 128, 64],
        'activation': nn.ELU
    },
    '64_sigmoid': {
        'batch_size': 32,
        'learning_rate': 0.0022005,
        'lambda': 0.46184,
        'sizes': [512, 256, 128, 64],
        'activation': nn.Sigmoid
    }
}


def build_model(config):
    """Build encoder and decoder based on configuration."""
    input_dim = 28 * 28 + 10  # MNIST image + one-hot labels

    # Build encoder layers
    encoder_layers = []
    prev_size = input_dim

    # Add layers with activation except the last one
    for size in config['sizes'][:-1]:
        encoder_layers.extend([
            nn.Linear(prev_size, size),
            config['activation']()
        ])
        prev_size = size

    # Add final encoder layer WITHOUT activation
    encoder_layers.append(nn.Linear(prev_size, config['sizes'][-1]))

    # Build decoder layers
    decoder_layers = []
    sizes_reversed = config['sizes'][::-1]
    prev_size = sizes_reversed[0]

    # Add layers with activation except the last one
    for size in sizes_reversed[1:]:
        decoder_layers.extend([
            nn.Linear(prev_size, size),
            config['activation']()
        ])
        prev_size = size

    # Add final decoder layer WITHOUT activation
    decoder_layers.append(nn.Linear(prev_size, input_dim))

    return nn.Sequential(*encoder_layers), nn.Sequential(*decoder_layers)


def train_model(arch_name, num_iterations, sum_losses=True, num_epochs=15):
    """Train a single model variant."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = MODEL_CONFIGS[arch_name]

    # Initialize model components
    encoder, decoder = build_model(config)
    encoder, decoder = encoder.to(device), decoder.to(device)

    def autoencoder(x):
        encoded = encoder(x)
        decoded = decoder(encoded)
        return decoded

    # Training setup
    optimizer = optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=config['learning_rate']
    )

    # Get data loaders using your existing function
    train_loader, test_loader, _ = get_mnist_loaders(config['batch_size'])
    train_loss = 0

    # Training loop
    for epoch in range(num_epochs):
        encoder.train()
        decoder.train()
        train_loss = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.view(images.size(0), -1).to(device)  # Flatten images
            diffuse_labels = create_diffuse_one_hot(labels).to(device)

            # Initial state
            initial_state = torch.cat((images, diffuse_labels), dim=1)
            current_state = initial_state
            targets = torch.cat((images, torch.eye(10, device=device)[labels]), dim=1)

            iteration_losses = []

            # Iterations loop
            for iteration in range(num_iterations):
                current_state = autoencoder(current_state)

                # Loss calculation
                outputs_label_probs = F.softmax(current_state[:, -10:], dim=1)
                image_loss = F.mse_loss(current_state[:, :-10], targets[:, :-10])
                label_loss = F.kl_div(outputs_label_probs.log(), targets[:, -10:])

                iteration_loss = image_loss + config['lambda'] * label_loss
                iteration_losses.append(iteration_loss)

            # Aggregate losses
            total_batch_loss = sum(iteration_losses) if sum_losses else iteration_losses[-1]

            # Optimization step
            optimizer.zero_grad()
            total_batch_loss.backward()
            optimizer.step()

            train_loss += total_batch_loss.item()

            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], "
                      f"Loss: {total_batch_loss.item():.4f}")

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss / len(train_loader):.4f}')

    return encoder, decoder, train_loss


def save_models(encoder, decoder, train_loss, arch_name, num_iterations, sum_losses=True):
    """Save trained models with appropriate naming convention."""
    os.makedirs('models', exist_ok=True)
    loss_type = ""
    # Create model name
    if num_iterations == 1:
        model_name = f"auto_{arch_name}_{num_iterations}"
    else:
        loss_type = "sum" if sum_losses else "last"
        model_name = f"auto_{arch_name}_{num_iterations}_{loss_type}"

    config = MODEL_CONFIGS[arch_name]
    log_data(model_name, config['sizes'][-1], config['activation'], num_iterations, loss_type, config['lambda'],
             config['batch_size'], config['learning_rate'], train_loss)

    torch.save(encoder.state_dict(), f'models/encoder_{model_name}.pth')
    torch.save(decoder.state_dict(), f'models/decoder_{model_name}.pth')
    print(f"Saved models as encoder_{model_name}.pth and decoder_{model_name}.pth")


def log_data(model_name, smallest_layer, activation_function, num_iterations, loss_type, lambda_,
             batch_size, learning_rate, train_loss):
    log_entry = {
        'timestamp': datetime.now(),
        'model_name': model_name,
        'smallest_layer': smallest_layer,
        'activation_function': activation_function,
        'num_iterations': num_iterations,
        'loss_type': loss_type,
        'lambda': lambda_,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'train_loss': train_loss,
        'test_set_accuracy': np.nan,
        'validation_set_accuracy': np.nan,
        'test_loss': np.nan
    }

    df_new = pd.DataFrame([log_entry])

    try:
        # Try to read existing CSV file
        df_existing = pd.read_csv('autos_log.csv')
        # Append new data
        df_updated = pd.concat([df_existing, df_new], ignore_index=True)
    except FileNotFoundError:
        # If file doesn't exist, create new DataFrame
        df_updated = df_new

    # Save to CSV
    df_updated.to_csv('autos_log.csv', index=False)

    return df_updated


def train_all_models():
    """Train all model variants."""
    for arch_name in MODEL_CONFIGS:
        print(f"\nTraining models for architecture: {arch_name}")

        # Train single iteration version
        print(f"\nTraining single iteration version...")
        encoder, decoder, train_loss = train_model(arch_name, num_iterations=1)
        save_models(encoder, decoder, train_loss, arch_name, num_iterations=1)

        # Train multiple iteration versions
        for num_iterations in range(2, 11):
            print(f"\nTraining {num_iterations} iterations versions...")

            # Sum loss version
            print("Training sum loss version...")
            encoder, decoder, train_loss = train_model(arch_name, num_iterations, sum_losses=True)
            save_models(encoder, decoder, train_loss, arch_name, num_iterations, sum_losses=True)

            # Last loss version
            print("Training last loss version...")
            encoder, decoder, train_loss = train_model(arch_name, num_iterations, sum_losses=False)
            save_models(encoder, decoder, train_loss, arch_name, num_iterations, sum_losses=False)


if __name__ == "__main__":
    train_all_models()


import torch
from torch import optim
from torch import nn
import torch.nn.functional as F
import os
import sys
import wandb

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import get_mnist_loaders
from helper import create_diffuse_one_hot

print('running auto_64_sigmoid.py')

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Constants
image_dim = 28 * 28
num_classes = 10
input_dim = image_dim + num_classes

# Define sweep configuration
sweep_config = {
    'method': 'random',
    'metric': {
        'name': 'test_accuracy',
        'goal': 'maximize'
    },
    'parameters': {
        'learning_rate': {
            'min': 0.0001,
            'max': 0.01
        },
        'batch_size': {
            'values': [32, 64, 128]
        },
        'lambda_weight': {
            'min': 0.0,
            'max': 0.5
        },
        'epochs': {
            'value': 15
        }
    }
}


def train_sweep():
    # Initialize wandb
    wandb.init()
    config = wandb.config

    print(f"\n{'=' * 50}")
    print(f"Starting 64 Sigmoid sweep run")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Lambda weight: {config.lambda_weight}")
    print(f"{'=' * 50}\n")

    # Define the encoder network
    encoder = nn.Sequential(
        nn.Linear(input_dim, 512),
        nn.Sigmoid(),
        nn.Linear(512, 256),
        nn.Sigmoid(),
        nn.Linear(256, 128),
        nn.Sigmoid(),
        nn.Linear(128, 64)
    ).to(device)

    # Define the decoder network
    decoder = nn.Sequential(
        nn.Linear(64, 128),
        nn.Sigmoid(),
        nn.Linear(128, 256),
        nn.Sigmoid(),
        nn.Linear(256, 512),
        nn.Sigmoid(),
        nn.Linear(512, input_dim)
    ).to(device)

    def autoencoder(x):
        encoded = encoder(x)
        decoded = decoder(encoded)
        return decoded

    # Get data loaders
    train_loader, test_loader, _ = get_mnist_loaders(batch_size=config.batch_size)

    # Initialize optimizer
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()),
                           lr=config.learning_rate)

    # Training loop
    for epoch in range(config.epochs):
        encoder.train()
        decoder.train()
        running_loss = 0.0

        for i, (images, labels) in enumerate(train_loader):
            images = images.view(images.size(0), -1).to(device)

            # Create inputs and targets
            diffuse_labels = create_diffuse_one_hot(labels).to(device)
            inputs = torch.cat((images, diffuse_labels), dim=1)
            targets = torch.cat((images, torch.eye(num_classes)[labels].to(device)), dim=1)

            optimizer.zero_grad()
            outputs = autoencoder(inputs)

            # Calculate losses
            outputs_label_probs = F.softmax(outputs[:, image_dim:], dim=1)
            image_loss = F.mse_loss(outputs[:, :image_dim], targets[:, :image_dim])
            label_loss = F.kl_div(outputs_label_probs.log(), targets[:, image_dim:])

            loss = image_loss + config.lambda_weight * label_loss
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (i + 1) % 100 == 0:
                wandb.log({
                    'epoch': epoch,
                    'train_loss': running_loss / 100,
                    'image_loss': image_loss.item(),
                    'label_loss': label_loss.item()
                })
                running_loss = 0.0

        # Evaluation
        encoder.eval()
        decoder.eval()
        correct = 0
        total = 0
        test_loss = 0.0

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.view(images.size(0), -1).to(device)
                diffuse_labels = create_diffuse_one_hot(labels).to(device)
                inputs = torch.cat((images, diffuse_labels), dim=1)
                targets = torch.cat((images, torch.eye(num_classes)[labels].to(device)), dim=1)

                outputs = autoencoder(inputs)

                # Calculate losses
                outputs_label_probs = F.softmax(outputs[:, image_dim:], dim=1)
                image_loss = F.mse_loss(outputs[:, :image_dim], targets[:, :image_dim])
                label_loss = F.kl_div(outputs_label_probs.log(), targets[:, image_dim:])

                loss = image_loss + config.lambda_weight * label_loss
                test_loss += loss.item()

                # Use last 10 outputs for prediction
                _, predicted = outputs[:, -num_classes:].max(1)
                total += labels.size(0)
                correct += predicted.eq(labels.to(device)).sum().item()

                if total % 1000 == 0:  # Debugging prints
                    print(f"\nPredictions sample:")
                    print(f"Labels: {labels[:5]}")
                    print(f"Predicted: {predicted[:5]}")
                    print(f"Running accuracy: {100. * correct / total:.2f}%")

            accuracy = 100 * correct / total
            wandb.log({
                'epoch': epoch,
                'test_accuracy': accuracy,
                'test_loss': test_loss / len(test_loader)
            })


if __name__ == "__main__":
    # Initialize wandb
    wandb.login()

    # Create and run sweep
    sweep_id = wandb.sweep(sweep_config, project="auto_64_sigmoid")
    print("\nStarting sweep for 64 Sigmoid architecture")
    wandb.agent(sweep_id, train_sweep, count=40)
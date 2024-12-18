import torch
from torch import optim
from torch import nn
import torch.nn.functional as F
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import get_mnist_loaders
from helper import create_diffuse_one_hot
import wandb

print('running hadamard_sweep.py')

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

    print(f"\n{'='*50}")
    print(f"Starting Hadamard sweep run")
    print(f"Project: {wandb.run.project}")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Lambda weight: {config.lambda_weight}")
    print(f"{'='*50}\n")

    # Define the two function networks
    f1 = nn.Sequential(
        nn.Linear(input_dim, input_dim)
    ).to(device)

    f2 = nn.Sequential(
        nn.Linear(input_dim, input_dim)
    ).to(device)

    def autoencoder(x):
        return f1(x) * f2(x)  # Hadamard product

    # Get data with configurable batch size
    train_loader, test_loader, _ = get_mnist_loaders(batch_size=config.batch_size)

    # Initialize optimizer for both networks
    optimizer = optim.Adam(list(f1.parameters()) + list(f2.parameters()), lr=config.learning_rate)

    # Training loop
    for epoch in range(config.epochs):
        f1.train()
        f2.train()
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images = images.view(images.size(0), -1).to(device)

            # Create diffuse labels for input and one-hot for targets
            diffuse_labels = create_diffuse_one_hot(labels).to(device)
            target_labels = F.one_hot(labels, num_classes=num_classes).float().to(device)

            # Concatenate images with diffuse labels for input
            inputs = torch.cat((images, diffuse_labels), dim=1)

            optimizer.zero_grad()
            outputs = autoencoder(inputs)

            # Split outputs into image and label components
            output_images = outputs[:, :image_dim]
            output_labels = outputs[:, image_dim:]

            # Calculate losses
            image_loss = F.mse_loss(output_images, images)
            label_loss = F.kl_div(F.log_softmax(output_labels, dim=1), target_labels, reduction='batchmean')

            # Combined loss
            loss = image_loss + config.lambda_weight * label_loss
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (i + 1) % 100 == 0:
                wandb.log({
                    'epoch': epoch,
                    'train_loss': running_loss / 100,
                    'image_loss': image_loss.item(),
                    'label_loss': label_loss.item(),
                    'model_type': 'Hadamard'
                })
                running_loss = 0.0

        # Test after each epoch
        f1.eval()
        f2.eval()
        correct = 0
        total = 0
        test_loss = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.view(images.size(0), -1).to(device)
                diffuse_labels = create_diffuse_one_hot(labels).to(device)
                target_labels = F.one_hot(labels, num_classes=num_classes).float().to(device)

                inputs = torch.cat((images, diffuse_labels), dim=1)
                outputs = autoencoder(inputs)

                output_images = outputs[:, :image_dim]
                output_labels = outputs[:, image_dim:]

                # Calculate test losses
                image_loss = F.mse_loss(output_images, images)
                label_loss = F.kl_div(F.log_softmax(output_labels, dim=1), target_labels, reduction='batchmean')
                loss = image_loss + config.lambda_weight * label_loss
                test_loss += loss.item()

                # Calculate accuracy
                _, predicted = output_labels.max(1)
                total += labels.size(0)
                correct += (predicted == labels.to(device)).sum().item()

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
    sweep_id = wandb.sweep(sweep_config, project="hadamard_sweep")
    print("\nStarting Hadamard sweep")
    wandb.agent(sweep_id, train_sweep, count=40)
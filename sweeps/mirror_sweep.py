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

print('running mirror_sweep.py')

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Constants
image_dim = 28 * 28
num_classes = 10
input_dim = image_dim + num_classes

# Define sweep configurations
sweep_config_elu = {
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
        },
        'activation': {
            'value': 'ELU'
        }
    }
}

sweep_config_sigmoid = dict(sweep_config_elu)
sweep_config_sigmoid['parameters']['activation']['value'] = 'Sigmoid'


def train_sweep():
    # Initialize wandb
    wandb.init()
    config = wandb.config

    # Set up activation function
    activation_fn = {
        'ELU': nn.ELU(),
        'Sigmoid': nn.Sigmoid(),
    }[config.activation]

    # Model definition
    encoder_and_decoder = nn.Sequential(
        nn.Linear(input_dim, input_dim),
        activation_fn
    ).to(device)

    def autoencoder(x):
        return encoder_and_decoder(x)

    # Get data with configurable batch size
    train_loader, test_loader, _ = get_mnist_loaders(batch_size=config.batch_size)

    # Initialize optimizer
    optimizer = optim.Adam(encoder_and_decoder.parameters(), lr=config.learning_rate)

    # Training loop
    for epoch in range(config.epochs):
        encoder_and_decoder.train()
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
                    'label_loss': label_loss.item()
                })
                running_loss = 0.0

        # Test after each epoch
        encoder_and_decoder.eval()
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

    # Create and run ELU sweep
    sweep_id_elu = wandb.sweep(sweep_config_elu, project="mirror_sweep_elu")
    wandb.agent(sweep_id_elu, train_sweep, count=40)

    # Create and run Sigmoid sweep
    sweep_id_sigmoid = wandb.sweep(sweep_config_sigmoid, project="mirror_sweep_sigmoid")
    wandb.agent(sweep_id_sigmoid, train_sweep, count=40)


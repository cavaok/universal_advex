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

print('running autoencoder_sweep_sigmoid.py')

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Constants
image_dim = 28 * 28
num_classes = 10
input_dim = image_dim + num_classes

# Architecture definitions
ARCHITECTURES = {
    '512': [(512, False)],
    '256': [(512, True), (256, False)],
    '128': [(512, True), (256, True), (128, False)],
    '64': [(512, True), (256, True), (128, True), (64, False)]
}


def create_encoder_decoder(architecture_layers):
    # Build encoder
    encoder_layers = []
    prev_dim = input_dim
    for dim, add_activation in architecture_layers:
        encoder_layers.append(nn.Linear(prev_dim, dim))
        if add_activation:
            encoder_layers.append(nn.Sigmoid())
        prev_dim = dim

    # Build decoder (reverse of encoder)
    decoder_layers = []
    layers_reversed = [(input_dim, True)] + [(architecture_layers[i - 1][0], architecture_layers[i - 1][1])
                                             for i in range(len(architecture_layers) - 1, 0, -1)]
    prev_dim = architecture_layers[-1][0]  # Start from bottleneck dimension
    for dim, add_activation in layers_reversed:
        decoder_layers.append(nn.Linear(prev_dim, dim))
        if add_activation:
            decoder_layers.append(nn.Sigmoid())
        prev_dim = dim

    return nn.Sequential(*encoder_layers), nn.Sequential(*decoder_layers)


# Define sweep configuration template
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
    print(f"Starting Sigmoid sweep run")
    print(f"Project: {wandb.run.project}")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Lambda weight: {config.lambda_weight}")
    print(f"{'=' * 50}\n")

    # Get architecture layers based on project name
    bottleneck = wandb.run.project.split('_')[1]  # Extract '512', '256', etc.
    architecture_layers = ARCHITECTURES[bottleneck]

    # Create encoder and decoder
    encoder, decoder = create_encoder_decoder(architecture_layers)
    encoder = encoder.to(device)
    decoder = decoder.to(device)

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

            # Create diffuse labels for input and one-hot for targets
            diffuse_labels = create_diffuse_one_hot(labels).to(device)
            target_labels = F.one_hot(labels, num_classes=num_classes).float().to(device)

            inputs = torch.cat((images, diffuse_labels), dim=1)

            optimizer.zero_grad()
            outputs = autoencoder(inputs)

            # Split outputs
            output_images = outputs[:, :image_dim]
            output_labels = outputs[:, image_dim:]

            # Calculate losses
            image_loss = F.mse_loss(output_images, images)
            label_loss = F.kl_div(F.log_softmax(output_labels, dim=1),
                                  target_labels, reduction='batchmean')

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
                    'activation': 'Sigmoid'
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
                target_labels = F.one_hot(labels, num_classes=num_classes).float().to(device)

                inputs = torch.cat((images, diffuse_labels), dim=1)
                outputs = autoencoder(inputs)

                output_images = outputs[:, :image_dim]
                output_labels = outputs[:, image_dim:]

                # Calculate test losses
                image_loss = F.mse_loss(output_images, images)
                label_loss = F.kl_div(F.log_softmax(output_labels, dim=1),
                                      target_labels, reduction='batchmean')
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

    # Run sweeps for each architecture
    for bottleneck in ['512', '256', '128', '64']:
        project_name = f"auto_{bottleneck}_sigmoid"
        sweep_id = wandb.sweep(sweep_config, project=project_name)
        print(f"\nStarting sweep for {project_name}")
        wandb.agent(sweep_id, train_sweep, count=40)


import torch
from torch import optim
from torch import nn
import torch.nn.functional as F
import os
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('/home/okcava/projects/universal_advex')
from data import get_mnist_loaders
import wandb

print('running mlp_sweep.py')

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        'epochs': {
            'value': 15
        },
        'hidden_dims': {
            'values': [[512, 256], [256, 128], [512, 256, 128]]
        },
        'activation': {
            'value': 'ELU'
        }
    }
}

sweep_config_sigmoid = {
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
        'epochs': {
            'value': 15
        },
        'hidden_dims': {
            'values': [[512, 256], [256, 128], [512, 256, 128]]
        },
        'activation': {
            'value': 'Sigmoid'
        }
    }
}

image_dim = 28 * 28


class MLP(nn.Module):
    def __init__(self, hidden_dims, activation):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()

        # Dictionary to map activation names to functions
        activation_fn = {
            'ELU': nn.ELU(),
            'Sigmoid': nn.Sigmoid(),
        }[activation]

        # Build layers dynamically
        layers = []
        prev_dim = image_dim
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                activation_fn,
            ])
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, 10))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.flatten(x)
        return self.layers(x)


def train_sweep():
    # Initialize wandb
    wandb.init()
    config = wandb.config

    # Get data with configurable batch size
    train_loader, test_loader, _ = get_mnist_loaders(batch_size=config.batch_size)

    # Initialize model and optimizer with wandb config
    model = MLP(config.hidden_dims, config.activation).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # Training loop
    for epoch in range(config.epochs):
        model.train()
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            target = F.one_hot(labels, num_classes=10).float()

            optimizer.zero_grad()
            outputs = model(images)
            log_probs = F.log_softmax(outputs, dim=1)
            loss = F.kl_div(log_probs, target, reduction='batchmean')
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (i + 1) % 100 == 0:
                wandb.log({
                    'epoch': epoch,
                    'train_loss': running_loss / 100
                })
                running_loss = 0.0

        # Test after each epoch
        model.eval()
        correct = 0
        total = 0
        test_loss = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Calculate test loss
                target = F.one_hot(labels, num_classes=10).float()
                log_probs = F.log_softmax(outputs, dim=1)
                test_loss += F.kl_div(log_probs, target, reduction='batchmean').item()

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
    sweep_id_elu = wandb.sweep(sweep_config_elu, project="mlp_sweep_elu")
    wandb.agent(sweep_id_elu, train_sweep, count=5)   # Run 5 ELU experiments

    # Create and run Sigmoid sweep
    sweep_id_sigmoid = wandb.sweep(sweep_config_sigmoid, project="mlp_sweep_sigmoid")
    wandb.agent(sweep_id_sigmoid, train_sweep, count=5)   # Run 5 Sigmoid experiments

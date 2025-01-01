import torch
from torch import optim
from torch import nn
import torch.nn.functional as F
import os
import sys

# Add paths for module imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('/home/okcava/projects/universal_advex')
from data import get_mnist_loaders

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

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


def train_model(model_config):
    # Unpack configuration
    activation = model_config['activation']
    learning_rate = model_config['learning_rate']
    batch_size = model_config['batch_size']
    hidden_dims = model_config['hidden_dims']
    epochs = model_config['epochs']

    # Setup model name and path
    model_name = f"mlp_{activation.lower()}_{'_'.join(map(str, hidden_dims))}.pth"
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, model_name)

    # Initialize model and training
    model = MLP(hidden_dims, activation).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train_loader, test_loader, _ = get_mnist_loaders(batch_size=batch_size)

    print(f"\nTraining {activation} MLP:")
    print(f"Learning rate: {learning_rate}")
    print(f"Batch size: {batch_size}")
    print(f"Hidden dimensions: {hidden_dims}")

    for epoch in range(epochs):
        # Training phase
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
                print(f'Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(train_loader)}], '
                      f'Loss: {running_loss / 100:.4f}')
                running_loss = 0.0

        # Evaluation phase
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
        avg_test_loss = test_loss / len(test_loader)

        print(f'Epoch [{epoch + 1}/{epochs}], '
              f'Test Accuracy: {accuracy:.2f}%, '
              f'Test Loss: {avg_test_loss:.4f}')

    # Save final model
    torch.save(model.state_dict(), model_path)
    print(f"\nTraining completed for {activation} MLP")
    print(f"Final accuracy: {accuracy:.2f}%")
    print(f"Model saved at: {model_path}")
    return accuracy


if __name__ == "__main__":
    # Model configurations
    configurations = [
        {
            'activation': 'Sigmoid',
            'learning_rate': 0.0017544,
            'batch_size': 32,
            'hidden_dims': [512, 256, 128],
            'epochs': 15
        },
        {
            'activation': 'ELU',
            'learning_rate': 0.0002456,
            'batch_size': 32,
            'hidden_dims': [512, 256, 128],
            'epochs': 15
        }
    ]

    # Train all models
    results = {}
    for config in configurations:
        print(f"\n{'=' * 50}")
        print(f"Training {config['activation']} MLP")
        print(f"{'=' * 50}")
        accuracy = train_model(config)
        results[config['activation']] = accuracy

    # Print final results
    print("\nFinal Results:")
    print("=" * 30)
    for activation, accuracy in results.items():
        print(f"{activation} MLP: {accuracy:.2f}%")
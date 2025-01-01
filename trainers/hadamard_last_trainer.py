import torch
from torch import optim
from torch import nn
import torch.nn.functional as F
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import get_mnist_loaders
from helper import create_diffuse_one_hot

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Constants
image_dim = 28 * 28
num_classes = 10
input_dim = image_dim + num_classes


def train_hadamard(num_iterations, config):
    """Train Hadamard network with specified number of iterations"""
    print(f"\n{'=' * 50}")
    print(f"Training Hadamard Network with {num_iterations} iterations (Last Loss)")
    print(f"Learning rate: {config['learning_rate']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Lambda weight: {config['lambda_weight']}")
    print(f"{'=' * 50}\n")

    # Define the two function networks
    f1 = nn.Sequential(
        nn.Linear(input_dim, input_dim)
    ).to(device)

    f2 = nn.Sequential(
        nn.Linear(input_dim, input_dim)
    ).to(device)

    def autoencoder(x):
        return f1(x) * f2(x)  # Hadamard product

    # Get data loaders
    train_loader, test_loader, _ = get_mnist_loaders(batch_size=config['batch_size'])

    # Initialize optimizer
    optimizer = optim.Adam(list(f1.parameters()) + list(f2.parameters()), lr=config['learning_rate'])

    # Training loop
    for epoch in range(config['epochs']):
        f1.train()
        f2.train()
        running_loss = 0.0

        for i, (images, labels) in enumerate(train_loader):
            images = images.view(images.size(0), -1).to(device)

            # Create diffuse labels for input and one-hot for targets
            diffuse_labels = create_diffuse_one_hot(labels).to(device)
            target_labels = F.one_hot(labels, num_classes=num_classes).float().to(device)

            # Initial state
            current_state = torch.cat((images, diffuse_labels), dim=1)

            optimizer.zero_grad()

            # Multiple iterations
            for iteration in range(num_iterations):
                current_state = autoencoder(current_state)

                # Only calculate and backpropagate loss for the final iteration
                if iteration == num_iterations - 1:
                    # Split outputs into image and label components
                    output_images = current_state[:, :image_dim]
                    output_labels = current_state[:, image_dim:]

                    # Calculate losses
                    image_loss = F.mse_loss(output_images, images)
                    label_loss = F.kl_div(F.log_softmax(output_labels, dim=1), target_labels, reduction='batchmean')

                    # Final loss
                    total_loss = image_loss + config['lambda_weight'] * label_loss
                    total_loss.backward()
                    optimizer.step()

                    running_loss += total_loss.item()

            if (i + 1) % 100 == 0:
                avg_loss = running_loss / 100
                print(f'Epoch [{epoch + 1}/{config["epochs"]}], '
                      f'Step [{i + 1}/{len(train_loader)}], '
                      f'Loss: {avg_loss:.4f}')
                running_loss = 0.0

        # Evaluation
        f1.eval()
        f2.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.view(images.size(0), -1).to(device)
                labels = labels.to(device)
                diffuse_labels = create_diffuse_one_hot(labels).to(device)

                # Initial state
                current_state = torch.cat((images, diffuse_labels), dim=1)

                # Multiple iterations
                for _ in range(num_iterations):
                    current_state = autoencoder(current_state)

                # Get final predictions
                output_labels = current_state[:, image_dim:]
                _, predicted = torch.max(output_labels, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f'Epoch [{epoch + 1}/{config["epochs"]}], Test Accuracy: {accuracy:.2f}%')

    # Save model
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, f'hadamard_last_iter{num_iterations}.pth')

    torch.save({
        'f1_state_dict': f1.state_dict(),
        'f2_state_dict': f2.state_dict()
    }, model_path)

    return accuracy


if __name__ == "__main__":
    # Configuration from sweep results
    config = {
        'learning_rate': 0.0014686,
        'batch_size': 128,
        'lambda_weight': 0.022089,
        'epochs': 15
    }

    # Train models with different iteration counts
    max_iterations = 10
    results = {}

    for num_iterations in range(1, max_iterations + 1):
        accuracy = train_hadamard(num_iterations, config)
        results[num_iterations] = accuracy

    # Print final results
    print("\nFinal Results:")
    print("=" * 50)
    for iterations, accuracy in results.items():
        print(f"Hadamard Last Loss with {iterations} iterations: {accuracy:.2f}%")
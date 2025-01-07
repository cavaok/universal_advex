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

# Single configuration
MODEL_CONFIG = {
    'batch_size': 128,
    'learning_rate': 0.0014686,
    'lambda': 0.022089
}


def train_model(num_iterations, num_epochs=15):
    """Train the model with specified number of iterations."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = 28 * 28 + 10  # MNIST image + one-hot labels

    # Model definition - Hadamard architecture
    f1 = nn.Sequential(
        nn.Linear(input_dim, input_dim)
    ).to(device)

    f2 = nn.Sequential(
        nn.Linear(input_dim, input_dim)
    ).to(device)

    # Print initial weight magnitudes
    print("\nInitial magnitudes:")
    print(f"f1 weight magnitude: {f1[0].weight.abs().mean()}")
    print(f"f2 weight magnitude: {f2[0].weight.abs().mean()}")

    def hadamard(x):
        return f1(x) * f2(x)

    # Training setup
    optimizer = optim.Adam(list(f1.parameters()) + list(f2.parameters()),
                           lr=MODEL_CONFIG['learning_rate'])

    train_loader, test_loader, _ = get_mnist_loaders(MODEL_CONFIG['batch_size'])
    train_loss = 0

    # Training loop
    for epoch in range(num_epochs):
        f1.train()
        f2.train()
        train_loss = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.view(images.size(0), -1).to(device)  # Flatten images
            diffuse_labels = create_diffuse_one_hot(labels).to(device)

            # Initial state
            initial_state = torch.cat((images, diffuse_labels), dim=1)
            current_state = initial_state

            if epoch == 0 and batch_idx == 0:
                print("\nFirst forward pass magnitudes:")
                print(f"Input magnitude: {initial_state.abs().mean()}")
                f1_out = f1(initial_state)
                f2_out = f2(initial_state)
                print(f"f1 output magnitude: {f1_out.abs().mean()}")
                print(f"f2 output magnitude: {f2_out.abs().mean()}")
                print(f"hadamard output magnitude: {(f1_out * f2_out).abs().mean()}")

            targets = torch.cat((images, torch.eye(10, device=device)[labels]), dim=1)
            iteration_losses = []

            # Iterations loop
            for iteration in range(num_iterations):
                current_state = hadamard(current_state)
                if epoch == 0 and batch_idx == 0:
                    print(f"\nIteration {iteration + 1} state magnitude: {current_state.abs().mean()}")

                # Loss calculation
                output_images = current_state[:, :-10]
                output_labels = current_state[:, -10:]
                image_loss = F.mse_loss(output_images, targets[:, :-10])
                label_loss = F.kl_div(F.log_softmax(output_labels, dim=1), targets[:, -10:], reduction='batchmean')

                iteration_loss = image_loss + MODEL_CONFIG['lambda'] * label_loss
                if epoch == 0 and batch_idx == 0:
                    print(f"Iteration {iteration + 1} loss: {iteration_loss.item()}")
                iteration_losses.append(iteration_loss)

            # Sum losses for all iterations
            total_batch_loss = torch.sum(torch.stack(iteration_losses))

            # Optimization step
            optimizer.zero_grad()
            total_batch_loss.backward()

            if epoch == 0 and batch_idx == 0:
                print("\nGradient magnitudes:")
                print(f"f1 weight gradients: {f1[0].weight.grad.abs().mean()}")
                print(f"f2 weight gradients: {f2[0].weight.grad.abs().mean()}")

            optimizer.step()

            if epoch == 0 and batch_idx == 0:
                print("\nPost-update weight magnitudes:")
                print(f"f1 weight magnitude: {f1[0].weight.abs().mean()}")
                print(f"f2 weight magnitude: {f2[0].weight.abs().mean()}")

            train_loss += total_batch_loss.detach()

            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], "
                      f"Loss: {total_batch_loss.detach():.4f}")

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss / len(train_loader):.4f}')

    return (f1, f2), train_loss


def save_model(models, train_loss, num_iterations):
    """Save trained models with appropriate naming convention."""
    os.makedirs('models', exist_ok=True)
    model_name = f"hadamard_{num_iterations}"
    f1, f2 = models

    log_data(model_name, train_loss, num_iterations)
    torch.save(f1.state_dict(), f'models/f1_{model_name}.pth')
    torch.save(f2.state_dict(), f'models/f2_{model_name}.pth')
    print(f"Saved models as f1_{model_name}.pth and f2_{model_name}.pth")


def log_data(model_name, train_loss, num_iterations):
    log_entry = {
        'timestamp': datetime.now(),
        'model_name': model_name,
        'num_iterations': num_iterations,
        'lambda': MODEL_CONFIG['lambda'],
        'batch_size': MODEL_CONFIG['batch_size'],
        'learning_rate': MODEL_CONFIG['learning_rate'],
        'train_loss': train_loss,
        'test_set_accuracy': np.nan,
        'validation_set_accuracy': np.nan,
        'test_loss': np.nan
    }

    df_new = pd.DataFrame([log_entry])

    try:
        df_existing = pd.read_csv('hadamard_log.csv')
        df_updated = pd.concat([df_existing, df_new], ignore_index=True)
    except FileNotFoundError:
        df_updated = df_new

    df_updated.to_csv('hadamard_log.csv', index=False)
    return df_updated


def train_all_iterations():
    """Train models for iterations 1-10."""
    for num_iterations in range(1, 11):
        print(f"\nTraining model with {num_iterations} iterations...")
        models, train_loss = train_model(num_iterations)
        save_model(models, train_loss, num_iterations)


if __name__ == "__main__":
    train_all_iterations()

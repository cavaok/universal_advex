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
    'batch_size': 32,
    'learning_rate': 0.001472,
    'lambda': 0.069189
}


def train_model(num_iterations, num_epochs=15):
    """Train the model with specified number of iterations."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = 28 * 28 + 10  # MNIST image + one-hot labels

    # Model definition
    encoder_and_decoder = nn.Sequential(
        nn.Linear(input_dim, input_dim),
        nn.ELU()
    ).to(device)

    def autoencoder(x):
        return encoder_and_decoder(x)

    # Training setup
    optimizer = optim.Adam(encoder_and_decoder.parameters(), lr=MODEL_CONFIG['learning_rate'])
    train_loader, test_loader, _ = get_mnist_loaders(MODEL_CONFIG['batch_size'])
    train_loss = 0

    # Training loop
    for epoch in range(num_epochs):
        encoder_and_decoder.train()
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

                iteration_loss = image_loss + MODEL_CONFIG['lambda'] * label_loss
                iteration_losses.append(iteration_loss)

            # Sum losses for all iterations
            total_batch_loss = torch.sum(torch.stack(iteration_losses))

            # Optimization step
            optimizer.zero_grad()
            total_batch_loss.backward()
            optimizer.step()

            train_loss += total_batch_loss.detach()

            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], "
                      f"Loss: {total_batch_loss.detach():.4f}")

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss / len(train_loader):.4f}')

    return encoder_and_decoder, train_loss


def save_model(model, train_loss, num_iterations):
    """Save trained model with appropriate naming convention."""
    os.makedirs('models', exist_ok=True)
    model_name = f"794_elu_{num_iterations}"

    log_data(model_name, train_loss, num_iterations)
    torch.save(model.state_dict(), f'models/model_{model_name}.pth')
    print(f"Saved model as model_{model_name}.pth")


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
        df_existing = pd.read_csv('794_log.csv')
        df_updated = pd.concat([df_existing, df_new], ignore_index=True)
    except FileNotFoundError:
        df_updated = df_new

    df_updated.to_csv('794_log.csv', index=False)
    return df_updated


def train_all_iterations():
    """Train models for iterations 1-10."""
    for num_iterations in range(1, 11):
        print(f"\nTraining model with {num_iterations} iterations...")
        model, train_loss = train_model(num_iterations)
        save_model(model, train_loss, num_iterations)


if __name__ == "__main__":
    train_all_iterations()
import torch
from torch import nn
import torch.nn.functional as F
import os
import sys
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('/home/okcava/projects/universal_advex')
from helper import create_diffuse_one_hot
from data import get_mnist_train_val_test_loaders


class ModelArchitecture:
    def __init__(self, input_dim, activation_type):
        self.input_dim = input_dim
        self.activation = nn.ELU if activation_type == 'elu' else nn.Sigmoid

    def build_512(self):
        encoder = nn.Sequential(
            nn.Linear(self.input_dim, 512),
            self.activation()
        )
        decoder = nn.Sequential(
            nn.Linear(512, self.input_dim)
        )
        return encoder, decoder

    def build_256(self):
        encoder = nn.Sequential(
            nn.Linear(self.input_dim, 512),
            self.activation(),
            nn.Linear(512, 256),
        )
        decoder = nn.Sequential(
            nn.Linear(256, 512),
            self.activation(),
            nn.Linear(512, self.input_dim)
        )
        return encoder, decoder

    def build_128(self):
        encoder = nn.Sequential(
            nn.Linear(self.input_dim, 512),
            self.activation(),
            nn.Linear(512, 256),
            self.activation(),
            nn.Linear(256, 128)
        )
        decoder = nn.Sequential(
            nn.Linear(128, 256),
            self.activation(),
            nn.Linear(256, 512),
            self.activation(),
            nn.Linear(512, self.input_dim)
        )
        return encoder, decoder

    def build_64(self):
        encoder = nn.Sequential(
            nn.Linear(self.input_dim, 512),
            self.activation(),
            nn.Linear(512, 256),
            self.activation(),
            nn.Linear(256, 128),
            self.activation(),
            nn.Linear(128, 64)
        )
        decoder = nn.Sequential(
            nn.Linear(64, 128),
            self.activation(),
            nn.Linear(128, 256),
            self.activation(),
            nn.Linear(256, 512),
            self.activation(),
            nn.Linear(512, self.input_dim)
        )
        return encoder, decoder


def evaluate_model(encoder, decoder, loader, device):
    """Evaluate model on given loader"""
    encoder.eval()
    decoder.eval()
    correct = 0
    total = 0
    total_loss = 0
    image_dim = 28 * 28

    with torch.no_grad():
        for images, labels in loader:
            images = images.view(images.size(0), -1).to(device)
            diffuse_labels = create_diffuse_one_hot(labels).to(device)
            inputs = torch.cat((images, diffuse_labels), dim=1)
            targets = torch.cat((images, torch.eye(10)[labels].to(device)), dim=1)

            # Forward pass
            encoded = encoder(inputs)
            outputs = decoder(encoded)

            # Calculate loss
            loss = F.mse_loss(outputs, targets)
            total_loss += loss.item()

            # Calculate accuracy
            _, predicted = outputs[:, image_dim:].max(1)
            total += labels.size(0)
            correct += predicted.eq(labels.to(device)).sum().item()

    accuracy = 100 * correct / total
    avg_loss = total_loss / len(loader)
    return accuracy, avg_loss


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = 28 * 28 + 10
    _, val_loader, test_loader = get_mnist_train_val_test_loaders(batch_size=128)
    models_dir = 'models'

    # Read existing CSV
    df = pd.read_csv('autos_log.csv')

    # For each encoder file in models directory
    encoder_files = [f for f in os.listdir(models_dir) if f.startswith('encoder_')]

    for encoder_file in encoder_files:
        # Get corresponding decoder file
        decoder_file = 'decoder_' + encoder_file[8:]  # Remove 'encoder_' prefix

        if decoder_file not in os.listdir(models_dir):
            print(f"Warning: No matching decoder found for {encoder_file}")
            continue

        # Extract model details from filename
        base_name = encoder_file[8:-4]  # Remove 'encoder_' and '.pth'
        parts = base_name.split('_')
        size = parts[1]
        activation = parts[2]

        # Build model architecture
        builder = ModelArchitecture(input_dim, activation)
        build_method = getattr(builder, f"build_{size}")
        encoder, decoder = build_method()
        encoder, decoder = encoder.to(device), decoder.to(device)

        # Load weights
        encoder.load_state_dict(torch.load(os.path.join(models_dir, encoder_file)))
        decoder.load_state_dict(torch.load(os.path.join(models_dir, decoder_file)))

        # Find matching row in DataFrame
        matching_rows = df[df['model_name'].str.contains(base_name, na=False)]
        if len(matching_rows) == 0:
            print(f"Warning: No matching log entry found for {base_name}")
            continue

        idx = matching_rows.index[0]

        # Evaluate on validation set
        val_accuracy, val_loss = evaluate_model(encoder, decoder, val_loader, device)

        # Evaluate on test set
        test_accuracy, test_loss = evaluate_model(encoder, decoder, test_loader, device)

        # Update DataFrame
        df.loc[idx, 'validation_set_accuracy'] = val_accuracy
        df.loc[idx, 'test_set_accuracy'] = test_accuracy
        df.loc[idx, 'test_loss'] = test_loss

        print(f"\nEvaluated {base_name}:")
        print(f"Validation Accuracy: {val_accuracy:.2f}%")
        print(f"Test Accuracy: {test_accuracy:.2f}%")
        print(f"Test Loss: {test_loss:.4f}")

    # Save updated CSV
    df.to_csv('autos_log.csv', index=False)
    print("\nEvaluation complete. Results saved to autos_log.csv")


if __name__ == "__main__":
    main()
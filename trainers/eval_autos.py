import torch
from torch import nn
import torch.nn.functional as F
import os
import sys
import pandas as pd
from collections import defaultdict
import re

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import get_mnist_loaders
from helper import create_diffuse_one_hot


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


def evaluate_model(encoder, decoder, test_loader, device):
    encoder.eval()
    decoder.eval()
    correct = 0
    total = 0
    total_image_loss = 0
    total_label_loss = 0
    image_dim = 28 * 28

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.view(images.size(0), -1).to(device)
            diffuse_labels = create_diffuse_one_hot(labels).to(device)
            inputs = torch.cat((images, diffuse_labels), dim=1)
            targets = torch.cat((images, torch.eye(10)[labels].to(device)), dim=1)

            # Forward pass
            encoded = encoder(inputs)
            outputs = decoder(encoded)

            # Calculate losses
            outputs_label_probs = F.softmax(outputs[:, image_dim:], dim=1)
            image_loss = F.mse_loss(outputs[:, :image_dim], targets[:, :image_dim])
            label_loss = F.kl_div(outputs_label_probs.log(), targets[:, image_dim:])

            # Classification accuracy
            _, predicted = outputs[:, -10:].max(1)
            total += labels.size(0)
            correct += predicted.eq(labels.to(device)).sum().item()

            total_image_loss += image_loss.item()
            total_label_loss += label_loss.item()

    accuracy = 100 * correct / total
    avg_image_loss = total_image_loss / len(test_loader)
    avg_label_loss = total_label_loss / len(test_loader)

    return accuracy, avg_image_loss, avg_label_loss


def parse_model_name(filename):
    pattern = r'auto_(\d+)_(\w+)_(\d+)(?:_(\w+))?'
    match = re.match(pattern, filename)
    if match:
        size, activation, iterations, loss_type = match.groups()
        return {
            'size': size,
            'activation': activation,
            'iterations': iterations,
            'loss_type': loss_type if loss_type else 'single'
        }
    return None


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _, test_loader, _ = get_mnist_loaders(batch_size=128)
    input_dim = 28 * 28 + 10
    results = []

    # Get all model files from models directory in trainers
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
    model_files = os.listdir(models_dir)
    encoder_files = [f for f in model_files if f.startswith('encoder_auto_')]

    for encoder_file in encoder_files:
        # Get corresponding decoder file
        decoder_file = 'decoder_' + encoder_file[8:]
        if decoder_file not in model_files:
            print(f"Warning: No matching decoder found for {encoder_file}")
            continue

        # Parse model details
        base_name = encoder_file[8:-4]  # Remove 'encoder_' prefix and '.pth' suffix
        details = parse_model_name(base_name)
        if not details:
            print(f"Warning: Could not parse model name {base_name}")
            continue

        print(f"\nEvaluating model: {base_name}")

        # Build appropriate architecture
        builder = ModelArchitecture(input_dim, details['activation'])
        build_method = getattr(builder, f"build_{details['size']}")
        encoder, decoder = build_method()
        encoder, decoder = encoder.to(device), decoder.to(device)

        # Load weights
        encoder.load_state_dict(torch.load(os.path.join(models_dir, encoder_file)))
        decoder.load_state_dict(torch.load(os.path.join(models_dir, decoder_file)))

        # Evaluate
        accuracy, image_loss, label_loss = evaluate_model(encoder, decoder, test_loader, device)

        # Store results
        results.append({
            'model': base_name,
            'size': details['size'],
            'activation': details['activation'],
            'iterations': details['iterations'],
            'loss_type': details['loss_type'],
            'accuracy': accuracy,
            'image_loss': image_loss,
            'label_loss': label_loss
        })

        print(f"Accuracy: {accuracy:.2f}%")
        print(f"Image Loss: {image_loss:.4f}")
        print(f"Label Loss: {label_loss:.4f}")

    # Create DataFrame and save results
    df = pd.DataFrame(results)
    results_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model_evaluation_results.csv')
    df.to_csv(results_file, index=False)
    print(f"\nResults saved to {results_file}")

    # Print summary statistics
    print("\nSummary by architecture:")
    for size in df['size'].unique():
        for activation in df['activation'].unique():
            subset = df[(df['size'] == size) & (df['activation'] == activation)]
            print(f"\n{size}_{activation}:")
            print(f"Average accuracy: {subset['accuracy'].mean():.2f}%")
            print(f"Best accuracy: {subset['accuracy'].max():.2f}%")
            print(f"Best model: {subset.loc[subset['accuracy'].idxmax(), 'model']}")


if __name__ == "__main__":
    main()
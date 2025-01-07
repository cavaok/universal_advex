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


def evaluate_model(f1, f2, loader, device):
    """Evaluate hadamard model on given loader"""
    f1.eval()
    f2.eval()
    correct = 0
    total = 0
    total_loss = 0
    image_dim = 28 * 28

    with torch.no_grad():
        for images, labels in loader:
            images = images.view(images.size(0), -1).to(device)
            diffuse_labels = create_diffuse_one_hot(labels).to(device)
            inputs = torch.cat((images, diffuse_labels), dim=1)
            target_labels = F.one_hot(labels, num_classes=10).float().to(device)

            # Forward pass using hadamard product
            outputs = f1(inputs) * f2(inputs)

            # Split outputs for loss and accuracy
            output_images = outputs[:, :image_dim]
            output_labels = outputs[:, image_dim:]

            # Calculate loss
            image_loss = F.mse_loss(output_images, images)
            label_loss = F.kl_div(F.log_softmax(output_labels, dim=1), target_labels, reduction='batchmean')
            loss = image_loss + 0.022089 * label_loss  # Using lambda from config
            total_loss += loss.item()

            # Calculate accuracy
            _, predicted = output_labels.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels.to(device)).sum().item()

    accuracy = 100 * correct / total
    avg_loss = total_loss / len(loader)
    return accuracy, avg_loss


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = 28 * 28 + 10
    _, val_loader, test_loader = get_mnist_train_val_test_loaders(batch_size=128)
    models_dir = 'hadamard_models'

    # Read existing CSV
    df = pd.read_csv('hadamard_log.csv')

    # For each f1 file in models directory
    f1_files = [f for f in os.listdir(models_dir) if f.startswith('f1_')]

    for f1_file in f1_files:
        # Get corresponding f2 file
        f2_file = 'f2_' + f1_file[3:]  # Replace 'f1_' with 'f2_'

        if f2_file not in os.listdir(models_dir):
            print(f"Warning: No matching f2 found for {f1_file}")
            continue

        # Extract model details from filename
        base_name = f1_file[3:-4]  # Remove 'f1_' and '.pth'

        # Create models
        f1 = nn.Sequential(nn.Linear(input_dim, input_dim)).to(device)
        f2 = nn.Sequential(nn.Linear(input_dim, input_dim)).to(device)

        # Load weights
        f1.load_state_dict(torch.load(os.path.join(models_dir, f1_file)))
        f2.load_state_dict(torch.load(os.path.join(models_dir, f2_file)))

        # Find matching row in DataFrame
        matching_rows = df[df['model_name'].str.contains(base_name, na=False)]
        if len(matching_rows) == 0:
            print(f"Warning: No matching log entry found for {base_name}")
            continue

        idx = matching_rows.index[0]

        # Evaluate on validation set
        val_accuracy, val_loss = evaluate_model(f1, f2, val_loader, device)

        # Evaluate on test set
        test_accuracy, test_loss = evaluate_model(f1, f2, test_loader, device)

        # Update DataFrame
        df.loc[idx, 'validation_set_accuracy'] = val_accuracy
        df.loc[idx, 'test_set_accuracy'] = test_accuracy
        df.loc[idx, 'test_loss'] = test_loss

        print(f"\nEvaluated {base_name}:")
        print(f"Validation Accuracy: {val_accuracy:.2f}%")
        print(f"Test Accuracy: {test_accuracy:.2f}%")
        print(f"Test Loss: {test_loss:.4f}")

    # Save updated CSV
    df.to_csv('hadamard_log.csv', index=False)
    print("\nEvaluation complete. Results saved to hadamard_log.csv")


if __name__ == "__main__":
    main()
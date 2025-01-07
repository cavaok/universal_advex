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


def evaluate_model(model, loader, device):
    """Evaluate model on given loader"""
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    image_dim = 28 * 28

    with torch.no_grad():
        for images, labels in loader:
            images = images.view(images.size(0), -1).to(device)
            diffuse_labels = create_diffuse_one_hot(labels).to(device)
            inputs = torch.cat((images, diffuse_labels), dim=1)
            targets = torch.cat((images, torch.eye(10, device=device)[labels]), dim=1)

            # Forward pass
            outputs = model(inputs)

            # Split outputs for loss calculation
            output_images = outputs[:, :-10]
            output_labels = outputs[:, -10:]

            # Calculate loss
            image_loss = F.mse_loss(output_images, targets[:, :-10])
            label_loss = F.kl_div(F.log_softmax(output_labels, dim=1), targets[:, -10:], reduction='batchmean')
            loss = image_loss + 0.069189 * label_loss  # Using lambda from config
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
    _, val_loader, test_loader = get_mnist_train_val_test_loaders(batch_size=32)
    models_dir = '794_models'

    # Read existing CSV
    df = pd.read_csv('794_log.csv')

    # For each model file in models directory
    model_files = [f for f in os.listdir(models_dir) if f.startswith('model_')]

    for model_file in model_files:
        # Extract model details from filename
        base_name = model_file[6:-4]  # Remove 'model_' and '.pth'

        # Create model
        model = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ELU()
        ).to(device)

        # Load weights
        model.load_state_dict(torch.load(os.path.join(models_dir, model_file)))

        # Find matching row in DataFrame
        matching_rows = df[df['model_name'].str.contains(base_name, na=False)]
        if len(matching_rows) == 0:
            print(f"Warning: No matching log entry found for {base_name}")
            continue

        idx = matching_rows.index[0]

        # Evaluate on validation set
        val_accuracy, val_loss = evaluate_model(model, val_loader, device)

        # Evaluate on test set
        test_accuracy, test_loss = evaluate_model(model, test_loader, device)

        # Update DataFrame
        df.loc[idx, 'validation_set_accuracy'] = val_accuracy
        df.loc[idx, 'test_set_accuracy'] = test_accuracy
        df.loc[idx, 'test_loss'] = test_loss

        print(f"\nEvaluated {base_name}:")
        print(f"Validation Accuracy: {val_accuracy:.2f}%")
        print(f"Test Accuracy: {test_accuracy:.2f}%")
        print(f"Test Loss: {test_loss:.4f}")

    # Save updated CSV
    df.to_csv('794_log.csv', index=False)
    print("\nEvaluation complete. Results saved to 794_log.csv")


if __name__ == "__main__":
    main()
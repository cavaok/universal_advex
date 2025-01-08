import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


# Loaders used for adversarial example training (function returns dictionary of loaders)
# ex. digit_loaders[8] returns the loader for images of 8s
def get_mnist_digit_loaders(batch_size=1):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    digit_loaders = {}

    for digit in range(10):
        indices = [i for i, (_, label) in enumerate(dataset.data) if label == digit]  # gets digit
        subset = Subset(dataset, indices)  # creates subset
        loader = DataLoader(subset, batch_size=batch_size, shuffle=False)  # creates data loader
        digit_loaders[digit] = loader  # store in dictionary

    return digit_loaders


# Loaders used for training models
def get_mnist_loaders(batch_size=10):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    adversarial_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    return train_loader, test_loader, adversarial_loader


# Loaders used for evaluating models
def get_mnist_train_val_test_loaders(batch_size=128, val_split=0.1):
    transform = transforms.Compose([transforms.ToTensor()])

    # Load datasets
    full_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # Split training into train/val
    train_size = int((1 - val_split) * len(full_train))
    val_size = len(full_train) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_train, [train_size, val_size])

    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

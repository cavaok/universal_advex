from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms


def get_mnist_loaders():
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)
    adversarial_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    return train_loader, test_loader, adversarial_loader



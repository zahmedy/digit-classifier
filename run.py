import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_loaders(batch_size=64):
    # 1. Define transfor: converts image to tensor AND normlizes pixel values
    transform = transforms.Compose([
            transforms.ToTensor(),               # (H,W,C) -> Tensor of shape (1,28,28)
            transforms.Normalize((0.5,), (0.5,)) # scale pixels from [0,1] to [-1,1]
    ])

    # 2) Load training and test datasets
    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_ds  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    # 3) Wrap in DataLoaders (mini-batches)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


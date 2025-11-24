import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # ---- Convolution layers ----
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3)    # (1,28,28) -> (8,26,26)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3)   # (8,26,26) -> (16,24,24)

        # ---- Fully connected layers ---- 
        self.fc1 = nn.Linear(16 * 5 * 5, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        # Block 1: Conv -> ReLU --> MaxPool
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        # Block 2
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
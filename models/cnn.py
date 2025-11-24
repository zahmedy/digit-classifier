import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 8, 3)      # (1,32,32) -> (8,30,30)
        self.conv2 = nn.Conv2d(8, 16, 3)     # (8,15,15) after pool -> (16,13,13)

        self.fc1 = nn.Linear(16 * 6 * 6, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)   # -> (8,15,15)

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)   # -> (16,13,13)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

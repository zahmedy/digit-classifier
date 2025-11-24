import torch
import torch.nn.functional as F

def train(model, loader, optimizer, device="cpu"):
    model.train()
    total_loss = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        # 1) Forward pass
        preds = model(images)

        # 2) Loss
        loss = F.cross_entropy(preds, labels)

        # 3) Backprop
        optimizer.zero_grad()
        loss.backward()

        # 4) Update weights
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

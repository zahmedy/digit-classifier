# utils/run.py (top of file)
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


def get_loaders(batch_size=64):
    from torchvision import transforms

    digits = load_digits()
    X = digits.images       # (N, 8, 8)
    y = digits.target       # (N,)

    X = X.astype("float32") / 16.0

    X = torch.from_numpy(X).unsqueeze(1)  # (N,1,8,8)
    y = torch.from_numpy(y).long()

    # Resize to 32x32 for CNN
    resize = transforms.Resize((32, 32), antialias=True)

    X_resized = torch.stack([resize(img) for img in X])  # (N,1,32,32)

    X_train, X_test, y_train, y_test = train_test_split(
        X_resized, y, test_size=0.2, random_state=42, stratify=y
    )

    train_ds = TensorDataset(X_train, y_train)
    test_ds  = TensorDataset(X_test, y_test)

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False),
    )



from models.cnn import CNN
from utils.train import train
import os

def main():
    train_loader, test_loader = get_loaders()

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    model = CNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("Training...")
    for epoch in range(10):
        loss = train(model, train_loader, optimizer, device)
        print(f"Epoch {epoch+1}, Loss = {loss:.4f}")

    # Save model for inference
    os.makedirs("models", exist_ok=True)
    save_path = "models/cnn_mnist.pt"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    main()

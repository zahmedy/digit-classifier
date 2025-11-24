import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from models.cnn import CNN

def load_model(checkpoint_path, device="cpu"):
    model = CNN().to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()  # 🔹 important: turn off dropout, etc.
    return model

def get_one_test_sample():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    test_ds = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=True)

    images, labels = next(iter(test_loader))  # one batch of size 1
    return images, labels

def predict_digit(model, image, device="cpu"):
    model.eval()
    with torch.no_grad():  # 🔹 no gradients needed for inference
        image = image.to(device)
        logits = model(image)          # shape: (1, 10)
        probs = torch.softmax(logits, dim=1)
        pred_class = probs.argmax(dim=1).item()
        confidence = probs.max().item()
    return pred_class, confidence



if __name__ == "__main__":
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    checkpoint_path = "models/cnn_mnist.pt"

    model = load_model(checkpoint_path, device=device)
    images, labels = get_one_test_sample()

    pred, conf = predict_digit(model, images, device=device)

    print(f"True label: {labels.item()}")
    print(f"Predicted: {pred} (confidence {conf:.3f})")

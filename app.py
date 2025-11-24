import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
import numpy as np
from torchvision import transforms

from models.cnn import CNN

# 1) Device
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# 2) Load model once
@st.cache_resource
def load_model():
    model = CNN().to(DEVICE)
    state_dict = torch.load("models/cnn_mnist.pt", map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    return model

model = load_model()

st.title("Digit Classifier (MNIST CNN)")
st.write("Upload a **28x28** or larger image of a single handwritten digit (0–9).")

uploaded_file = st.file_uploader("Upload a digit image", type=["png", "jpg", "jpeg"])

import numpy as np
from PIL import Image
import torch
from torchvision import transforms

transform = transforms.Compose([
    transforms.Grayscale(),              # make sure it's 1-channel
    transforms.Resize((32, 32)),         # match training
    transforms.ToTensor(),               # convert to tensor in [0,1]
    # ❗ No normalization here — training data was 0..1, not normalized
])

def preprocess_image(img: Image.Image):
    img = img.convert("L")               # grayscale
    tensor = transform(img)              # (1, 32, 32)
    tensor = tensor.unsqueeze(0).to(DEVICE)  # (1, 1, 32, 32)
    return tensor


def predict_digit(image_tensor: torch.Tensor):
    with torch.no_grad():
        logits = model(image_tensor)   # shape: (1,10)
        probs = F.softmax(logits, dim=1)
        pred_class = probs.argmax(dim=1).item()
        confidence = probs.max().item()
    return pred_class, confidence

# with torch.no_grad():
#     img_for_display = tensor[0, 0].cpu().numpy()
# st.image(img_for_display, caption="Processed 32x32 image", width=150, clamp=True)


if uploaded_file is not None:
    orig_img = Image.open(uploaded_file)
    st.image(orig_img, caption="Original uploaded image", width=150)

    if st.button("Predict digit"):
        tensor = preprocess_image(orig_img)

        # show processed 28x28 image
        with torch.no_grad():
            img_for_display = tensor[0, 0].cpu().numpy()
            img_for_display = (img_for_display * 0.5) + 0.5  # [-1,1] -> [0,1]

        st.image(img_for_display, caption="Processed 28x28 image", width=150, clamp=True)

        pred, conf = predict_digit(tensor)
        st.markdown(f"### ✅ Predicted digit: **{pred}**")
        st.markdown(f"Confidence: `{conf:.3f}`")




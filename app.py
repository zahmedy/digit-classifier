from pathlib import Path
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from models.cnn import CNN

# Resolve project paths once
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "cnn_mnist.pt"

st.set_page_config(
    page_title="Digit Classifier",
    page_icon="🔢",
    layout="centered",
)

# Device selection keeps Apple Silicon fast while falling back to CPU elsewhere.
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


@st.cache_resource(show_spinner=True)
def load_model():
    if not MODEL_PATH.exists():
        st.error(f"Model weights not found at {MODEL_PATH}. Run `python run.py` to train.")
        st.stop()

    model = CNN().to(DEVICE)
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def preprocess_image(img: Image.Image) -> torch.Tensor:
    transform = transforms.Compose(
        [
            transforms.Grayscale(),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),  # keeps values in [0,1]
        ]
    )
    tensor = transform(img.convert("L"))  # (1, 32, 32)
    return tensor.unsqueeze(0).to(DEVICE)  # (1, 1, 32, 32)


def predict_digit(image_tensor: torch.Tensor):
    with torch.no_grad():
        logits = model(image_tensor)
        probs = F.softmax(logits, dim=1)
        best_idx = probs.argmax(dim=1).item()
        confidence = probs.max().item()
        top_probs, top_idxs = probs.topk(3)
    return best_idx, confidence, top_idxs.squeeze(0), top_probs.squeeze(0)


def show_processed_image(tensor: torch.Tensor):
    img_for_display = tensor[0, 0].detach().cpu().numpy()
    st.image(img_for_display, caption="Model input (32×32 grayscale)", width=150, clamp=True)


model = load_model()

st.title("Digit Classifier")
st.caption("Lightweight CNN trained on handwritten digits, resized to 32×32 grayscale.")

with st.sidebar:
    st.markdown("### How to get a good prediction")
    st.write(
        "- Use a single, centered digit on a clean background.\n"
        "- High contrast works best (dark background, light stroke).\n"
        "- The app will resize to 32×32 pixels before inference."
    )

uploaded_file = st.file_uploader("Upload a digit image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    orig_img = Image.open(uploaded_file)
    cols = st.columns(2)
    cols[0].image(orig_img, caption="Original upload", width=200)

    if st.button("Predict digit"):
        tensor = preprocess_image(orig_img)
        cols[1].image(orig_img.resize((32, 32)).convert("L"), caption="Resized preview", width=200)
        show_processed_image(tensor)

        pred, conf, top_idxs, top_probs = predict_digit(tensor)
        st.markdown(f"### ✅ Predicted digit: **{pred}**")
        st.write(f"Confidence: `{conf:.3f}`")

        st.markdown("Top 3 probabilities")
        prob_cols = st.columns(len(top_idxs))
        for col, idx, prob in zip(prob_cols, top_idxs, top_probs):
            col.metric(label=f"Digit {idx.item()}", value=f"{prob.item() * 100:.1f}%")


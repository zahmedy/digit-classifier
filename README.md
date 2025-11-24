# Digit Classifier

Lightweight PyTorch CNN trained on handwritten digits (scikit-learn digits dataset, upscaled to 32×32) and served through a Streamlit UI.

## Features
- Streamlit app with guided upload flow and confidence breakdown.
- Cached model loading with Apple Silicon (`mps`) or CPU fallback.
- One-command training script (`run.py`) that reports test accuracy and saves weights to `models/cnn_mnist.pt`.
- Clear project layout for quick portfolio walkthroughs.

## Quickstart
1) Create an environment and install dependencies:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
2) Launch the web app:
```bash
streamlit run app.py
```
3) Upload a single handwritten digit image (PNG/JPG). High contrast works best; the app will resize to 32×32 grayscale before inference.

## Train your own weights
```bash
python run.py
```
- Uses the scikit-learn digits dataset, resized to 32×32.
- Prints loss per epoch and final test accuracy.
- Saves the checkpoint to `models/cnn_mnist.pt`, which the Streamlit app loads automatically.

## Inference via CLI (optional)
```bash
python utils/infer.py
```
Loads the saved checkpoint and runs one random test sample for a sanity check.

## Project structure
```
app.py                 # Streamlit UI for uploading and predicting digits
run.py                 # Train/evaluate the CNN and save weights
models/cnn.py          # CNN architecture (2 conv layers + 2 FC layers)
models/cnn_mnist.pt    # Pretrained weights (created by run.py)
utils/train.py         # Training loop
utils/infer.py         # Small CLI inference helper
```

## Model & data
- Architecture: two 3×3 conv layers (8 and 16 channels) → max pool → two fully connected layers (64 → 10 classes) with ReLU activations.
- Input preprocessing: grayscale conversion, resize to 32×32, tensor values kept in `[0, 1]`.
- Dataset: scikit-learn digits (8×8) upscaled to match the model’s expected input.

## Tech stack
- PyTorch + Torchvision for modeling and preprocessing
- Streamlit for UI
- scikit-learn for the digits dataset

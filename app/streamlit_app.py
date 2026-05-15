"""
========================================
STREAMLIT CIFAR-10 DEMO APP
========================================

This app:

1. Loads production CNN_V6 model
2. Accepts image upload
3. Runs inference
4. Displays prediction in UI
"""

import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

# --------------------------------------------------
# IMPORTS
# --------------------------------------------------
import streamlit as st
from PIL import Image
import torch

from src.production_inference import load_production_model, predict_image
from torchvision import transforms

# --------------------------------------------------
# CLASS LABELS
# --------------------------------------------------
classes = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# --------------------------------------------------
# LOAD MODEL (CACHE FOR PERFORMANCE)
# --------------------------------------------------
@st.cache_resource
def get_model():
    return load_production_model()

model = get_model()

# --------------------------------------------------
# IMAGE TRANSFORM
# --------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

# --------------------------------------------------
# APP UI
# --------------------------------------------------
st.title("CIFAR-10 Image Classifier")
st.write("Upload an image and get a prediction using CNN_V6")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_container_width=True)

    image_tensor = transform(image).unsqueeze(0)

    pred_idx = predict_image(model, image_tensor)

    pred_class = classes[pred_idx]

    st.success(f"Prediction: {pred_class}")
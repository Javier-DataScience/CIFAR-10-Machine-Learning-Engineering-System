"""
========================================
FASTAPI CIFAR-10 INFERENCE SERVICE
========================================

This API:

1. Loads production model:
       models/best_model.pt

2. Accepts uploaded images

3. Runs CNN inference

4. Returns predicted CIFAR-10 class

========================================
"""

# --------------------------------------------------
# IMPORTS
# --------------------------------------------------
import io

import torch
from PIL import Image

from fastapi import FastAPI, UploadFile, File

from torchvision import transforms

from src.production_inference import (
    load_production_model,
    predict_image
)

# --------------------------------------------------
# FASTAPI APP
# --------------------------------------------------
app = FastAPI(
    title="CIFAR-10 CNN Inference API"
)

# --------------------------------------------------
# DEVICE
# --------------------------------------------------
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# --------------------------------------------------
# LOAD PRODUCTION MODEL
# --------------------------------------------------
model = load_production_model()

# --------------------------------------------------
# CIFAR-10 CLASSES
# --------------------------------------------------
classes = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]

# --------------------------------------------------
# IMAGE TRANSFORM
# --------------------------------------------------
transform = transforms.Compose([

    transforms.Resize((32, 32)),

    transforms.ToTensor()
])

# --------------------------------------------------
# ROOT ENDPOINT
# --------------------------------------------------
@app.get("/")
def root():

    return {
        "message":
        "CIFAR-10 FastAPI inference service is running"
    }

# --------------------------------------------------
# PREDICTION ENDPOINT
# --------------------------------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # ----------------------------------------------
    # READ IMAGE
    # ----------------------------------------------
    image_bytes = await file.read()

    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    # ----------------------------------------------
    # TRANSFORM IMAGE
    # ----------------------------------------------
    image_tensor = transform(image)

    image_tensor = image_tensor.unsqueeze(0)

    # ----------------------------------------------
    # RUN INFERENCE
    # ----------------------------------------------
    predicted_idx = predict_image(
        model,
        image_tensor
    )

    predicted_class = classes[predicted_idx]

    # ----------------------------------------------
    # RESPONSE
    # ----------------------------------------------
    return {

        "predicted_class_index":
        predicted_idx,

        "predicted_class_name":
        predicted_class
    }
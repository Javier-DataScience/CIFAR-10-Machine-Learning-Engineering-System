"""
========================================
PRODUCTION INFERENCE MODULE
========================================

This module:

1. Loads the production artifact:
       models/best_model.pt

2. Uses frozen production architecture:
       CNN_V6

3. Provides reusable production inference
   utilities for:
   - FastAPI
   - Docker
   - Streamlit

IMPORTANT:
- Experimental models are NOT used here
- Only production artifact is loaded
"""

# --------------------------------------------------
# IMPORTS
# --------------------------------------------------
import torch

from src.model_registry import get_model

# --------------------------------------------------
# DEVICE
# --------------------------------------------------
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# --------------------------------------------------
# PRODUCTION CONFIGURATION
# --------------------------------------------------
PRODUCTION_MODEL_NAME = "cnn_v6"

PRODUCTION_MODEL_PATH = "models/best_model.pt"

# --------------------------------------------------
# LOAD PRODUCTION MODEL
# --------------------------------------------------
def load_production_model():

    model = get_model(
        PRODUCTION_MODEL_NAME
    ).to(device)

    state_dict = torch.load(
        PRODUCTION_MODEL_PATH,
        map_location=device
    )

    model.load_state_dict(state_dict)

    model.eval()

    return model

# --------------------------------------------------
# SINGLE IMAGE PREDICTION
# --------------------------------------------------
def predict_image(model, image_tensor):

    image_tensor = image_tensor.to(device)

    with torch.no_grad():

        outputs = model(image_tensor)

        _, predicted = torch.max(outputs, 1)

    return predicted.item()

# --------------------------------------------------
# MAIN TEST
# --------------------------------------------------
if __name__ == "__main__":

    model = load_production_model()

    print(
        "Production model loaded successfully"
    )
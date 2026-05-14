# This module defines the MODEL REGISTRY.
# It ensures strict mapping between model names and architectures.
# This prevents architecture confusion during training and inference.

# This module enforces strict model version control.
# Every CNN version must be explicitly registered.

from src.models import CNN_V1, CNN_V2, CNN_V3, CNN_V4


MODEL_MAP = {
    "cnn_v1": CNN_V1,
    "cnn_v2": CNN_V2,
    "cnn_v3": CNN_V3,
    "cnn_v4": CNN_V4,
}


def get_model(model_name):
    """
    Returns model class based on registry key.
    Ensures strict control over architecture selection.
    """
    if model_name not in MODEL_MAP:
        raise ValueError(f"Model {model_name} not found in registry")

    return MODEL_MAP[model_name]()
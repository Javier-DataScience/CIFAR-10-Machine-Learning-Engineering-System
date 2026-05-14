# This module defines the MODEL REGISTRY.
# It ensures strict mapping between model names and architectures.
# This prevents architecture confusion during training and inference.

from src.models import CNN_V1


# -------------------------
# MODEL REGISTRY
# -------------------------
MODEL_MAP = {
    "cnn_v1": CNN_V1
    # Future models will be added here:
    # "cnn_v2": CNN_V2,
    # "cnn_v3": CNN_V3,
    # "cnn_v4": CNN_V4,
    # "cnn_v5": CNN_V5,
}


def get_model(model_name):
    """
    Returns model class based on registry key.
    """
    if model_name not in MODEL_MAP:
        raise ValueError(f"Model {model_name} not found in registry")

    return MODEL_MAP[model_name]()
# This module defines the MODEL REGISTRY.
# It ensures strict mapping between model names and architectures.
# This prevents architecture confusion during training and inference.

# This module enforces strict model version control.
# Every CNN version must be explicitly registered.

from src.models import (
    CNN_V1,
    CNN_V2,
    CNN_V3,
    CNN_V4,
    CNN_V5,
    CNN_V6
)


MODEL_MAP = {

    "cnn_v1": CNN_V1,
    "cnn_v2": CNN_V2,
    "cnn_v3": CNN_V3,
    "cnn_v4": CNN_V4,
    "cnn_v5": CNN_V5,
    "cnn_v6": CNN_V6,
}


def get_model(model_name):

    if model_name not in MODEL_MAP:

        raise ValueError(
            f"Model '{model_name}' not found in registry. "
            f"Available models: {list(MODEL_MAP.keys())}"
        )

    return MODEL_MAP[model_name]()
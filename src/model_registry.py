"""
========================================
MODEL REGISTRY (FINAL STABLE VERSION)
========================================

This module is the SINGLE SOURCE OF TRUTH
for selecting model architectures.

Rules:
- Every model MUST be explicitly registered
- Names MUST match training calls exactly
"""

from src.models import CNN_V1, CNN_V2


def get_model(model_name):

    MODEL_MAP = {
        "cnn_v1": CNN_V1(),
        "cnn_v2": CNN_V2(),
    }

    if model_name not in MODEL_MAP:
        raise ValueError(
            f"Model '{model_name}' not found in registry. "
            f"Available models: {list(MODEL_MAP.keys())}"
        )

    return MODEL_MAP[model_name]
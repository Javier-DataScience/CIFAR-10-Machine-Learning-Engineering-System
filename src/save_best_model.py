"""
========================================
SAVE BEST MODEL ARTIFACT
========================================

This module:

1. Loads the best-performing architecture (CNN_V6)
2. Loads frozen trained weights
3. Saves canonical production artifact:
       models/best_model.pt

4. Creates metadata file:
       models/best_model_metadata.txt

5. Registers model artifact in MLflow

Purpose:
- Separate experimental models from production model
- Create stable artifact for inference APIs
- Prepare deployment workflow
"""

# --------------------------------------------------
# IMPORTS
# --------------------------------------------------
import os
import torch
import mlflow
import mlflow.pytorch

from src.model_registry import get_model

# --------------------------------------------------
# DEVICE
# --------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------------------------
# BEST MODEL CONFIGURATION
# --------------------------------------------------
BEST_MODEL_NAME = "cnn_v6"

SOURCE_MODEL_PATH = "models/cnn_v6.pt"

BEST_MODEL_PATH = "models/best_model.pt"

TEST_ACCURACY = 0.7733

# --------------------------------------------------
# LOAD MODEL ARCHITECTURE
# --------------------------------------------------
model = get_model(BEST_MODEL_NAME).to(device)

# --------------------------------------------------
# LOAD TRAINED WEIGHTS
# --------------------------------------------------
state_dict = torch.load(
    SOURCE_MODEL_PATH,
    map_location=device
)

model.load_state_dict(state_dict)

model.eval()

print("Best model loaded successfully")

# --------------------------------------------------
# SAVE PRODUCTION ARTIFACT
# --------------------------------------------------
torch.save(
    model.state_dict(),
    BEST_MODEL_PATH
)

print(f"Production artifact saved at: {BEST_MODEL_PATH}")

# --------------------------------------------------
# CREATE METADATA FILE
# --------------------------------------------------
metadata = f"""
========================================
BEST MODEL METADATA
========================================

Model Name: {BEST_MODEL_NAME}

Source Artifact:
{SOURCE_MODEL_PATH}

Production Artifact:
{BEST_MODEL_PATH}

Test Accuracy:
{TEST_ACCURACY}

Purpose:
Production inference serving

========================================
"""

metadata_path = "models/best_model_metadata.txt"

with open(metadata_path, "w") as f:
    f.write(metadata)

print(f"Metadata file saved at: {metadata_path}")

# --------------------------------------------------
# REGISTER MODEL IN MLFLOW
# --------------------------------------------------
mlflow.set_experiment("CIFAR10_Production_Models")

with mlflow.start_run(run_name="best_model_cnn_v6"):

    mlflow.log_param("model_name", BEST_MODEL_NAME)

    mlflow.log_metric("test_accuracy", TEST_ACCURACY)

    mlflow.log_artifact(BEST_MODEL_PATH)

    mlflow.log_artifact(metadata_path)

    mlflow.pytorch.log_model(
        model,
        artifact_path="best_model"
    )

print("Best model registered in MLflow successfully")
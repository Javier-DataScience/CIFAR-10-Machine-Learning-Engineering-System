"""
========================================
TRAINING MODULE (CIFAR-10 PROJECT)
========================================

This module is responsible for:

1. Loading CIFAR-10 dataset (train/test)
2. Selecting model architecture from registry (CNN_V1, CNN_V2, etc.)
3. Training the model for a given number of epochs
4. Evaluating performance (train + test accuracy)
5. Logging metrics and parameters to MLflow
6. Saving a FROZEN model (.pt file)

IMPORTANT RULES:
- Each model is saved ONLY ONCE (no overwrite allowed)
- MLflow experiment name must remain constant
- Saved models are frozen for inference only
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import mlflow

from src.data_loader import get_cifar10_dataloaders
from src.model_registry import get_model


def train_model(model_name, epochs=5):

    print(f"\n🚀 TRAINING STARTED: {model_name}\n")

    # -----------------------------
    # MLflow CONFIGURATION
    # -----------------------------
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("cifar10_cnn_experiment")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = get_cifar10_dataloaders()
    model = get_model(model_name).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    with mlflow.start_run(run_name=model_name):

        mlflow.log_param("model_name", model_name)
        mlflow.log_param("epochs", epochs)

        # -----------------------------
        # TRAINING LOOP
        # -----------------------------
        for epoch in range(epochs):

            model.train()

            running_loss = 0.0
            correct = 0
            total = 0

            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            train_acc = correct / total
            avg_loss = running_loss / len(train_loader)

            # -----------------------------
            # TEST EVALUATION
            # -----------------------------
            model.eval()
            correct = 0
            total = 0

            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)

                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            test_acc = correct / total

            # MLflow logging
            mlflow.log_metric("loss", avg_loss, step=epoch)
            mlflow.log_metric("train_accuracy", train_acc, step=epoch)
            mlflow.log_metric("test_accuracy", test_acc, step=epoch)

            print(
                f"Epoch {epoch+1}/{epochs} | "
                f"Loss: {avg_loss:.4f} | "
                f"Train Acc: {train_acc:.4f} | "
                f"Test Acc: {test_acc:.4f}"
            )

    # -----------------------------
    # MODEL FREEZING + SAFE SAVE
    # -----------------------------
    os.makedirs("models", exist_ok=True)

    save_path = f"models/{model_name}.pt"

    # Prevent accidental overwrite
    if os.path.exists(save_path):
        raise RuntimeError(
            f"\n❌ MODEL ALREADY EXISTS: {save_path}\n"
            "Delete it manually if retraining is intended.\n"
        )

    # Freeze model (inference mode)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    torch.save(model.state_dict(), save_path)

    print(f"\n✅ MODEL FROZEN AND SAVED: {save_path}\n")


# -----------------------------
# TERMINAL ENTRY POINT
# -----------------------------
if __name__ == "__main__":

    # CHANGE ONLY THIS LINE WHEN SWITCHING MODELS
    train_model("cnn_v2", epochs=5)
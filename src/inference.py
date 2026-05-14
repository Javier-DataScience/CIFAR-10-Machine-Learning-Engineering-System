# This module handles:
# 1. Loading frozen CNN models (.pt)
# 2. Running evaluation on CIFAR-10 test set
# 3. Returning test accuracy
# 4. Ensures consistent evaluation across all model versions

import torch
from src.data_loader import get_cifar10_dataloaders
from src.models import CNN_V1


def load_model(model_class, model_path, device="cpu"):
    """
    Loads a frozen model from disk.
    """
    model = model_class().to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def evaluate_model(model, testloader, device="cpu"):
    """
    Evaluates model on test dataset and returns accuracy.
    """
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total


def run_frozen_model_evaluation(model_class, model_path):
    """
    Full pipeline:
    load → evaluate → return accuracy
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, testloader = get_cifar10_dataloaders(batch_size=64)

    model = load_model(model_class, model_path, device)
    acc = evaluate_model(model, testloader, device)

    return acc
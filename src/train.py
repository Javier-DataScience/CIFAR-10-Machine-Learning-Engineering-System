# This script trains CNN_V1 on CIFAR-10 and saves a frozen model artifact.
# It follows strict ML system rules:
# - no MLflow dependency
# - no notebook logic
# - frozen model is saved only once (no overwrite)
# - clean epoch logging: loss, train accuracy, test accuracy in one line

import torch
import torch.nn as nn
import torch.optim as optim
import os

from src.models import CNN_V1
from src.data_loader import get_cifar10_dataloaders


def train_cnn_v1(epochs=5, lr=0.001):

    # -------------------------
    # DEVICE SETUP
    # -------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------
    # MODEL
    # -------------------------
    model = CNN_V1().to(device)

    # -------------------------
    # DATA
    # -------------------------
    trainloader, testloader = get_cifar10_dataloaders(batch_size=64)

    # -------------------------
    # LOSS + OPTIMIZER
    # -------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # -------------------------
    # TRAINING LOOP
    # -------------------------
    for epoch in range(epochs):

        model.train()

        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_acc = correct_train / total_train

        # -------------------------
        # TEST EVALUATION
        # -------------------------
        model.eval()

        correct_test = 0
        total_test = 0

        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                _, predicted = torch.max(outputs, 1)

                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()

        test_acc = correct_test / total_test

        # -------------------------
        # CLEAN ONE-LINE LOG
        # -------------------------
        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Loss: {running_loss/len(trainloader):.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Test Acc: {test_acc:.4f}"
        )

    # -------------------------
    # SAVE FROZEN MODEL (NO OVERWRITE)
    # -------------------------
    os.makedirs("models", exist_ok=True)

    model_path = "models/cnn_v1.pt"

    if not os.path.exists(model_path):
        torch.save(model.state_dict(), model_path)
        print(f"\nSaved frozen model -> {model_path}")
    else:
        print(f"\nModel already exists (FROZEN): {model_path} - skipping overwrite")


if __name__ == "__main__":
    train_cnn_v1()
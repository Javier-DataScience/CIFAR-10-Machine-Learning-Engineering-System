# This file defines all CNN architectures for the project.
# CNN_V1 is a frozen baseline architecture for CIFAR-10 classification.
# Once defined, CNN_V1 must NEVER be modified. Any changes require a new version (CNN_V2, CNN_V3, etc.)

import torch
import torch.nn as nn


class CNN_V1(nn.Module):
    """
    Frozen baseline CNN for CIFAR-10 classification.

    Input:
        - Tensor shape: (batch_size, 3, 32, 32)

    Output:
        - Tensor shape: (batch_size, 10) logits
    """

    def __init__(self):
        super(CNN_V1, self).__init__()

        # Feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
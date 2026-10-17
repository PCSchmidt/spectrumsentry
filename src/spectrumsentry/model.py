"""A small spectrogram CNN."""

from __future__ import annotations

import torch
from torch import nn


class SpecCnn(nn.Module):
    def __init__(self, n_classes: int, width: int = 32, in_channels: int = 1):
        super().__init__()

        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))

        self.features = nn.Sequential(block(in_channels, width), block(width, 2 * width), block(2 * width, 4 * width), block(4 * width, 4 * width))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.2), nn.Linear(4 * width, n_classes))

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Penultimate features (B, 4*width): global-average-pooled convolutional output."""
        return self.head[1](self.head[0](self.features(x)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))

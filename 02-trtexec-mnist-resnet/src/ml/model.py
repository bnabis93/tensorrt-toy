"""Basic CNN model
from https://github.com/pytorch/etensoramples/blob/main/mnist/main.py
- Author: bono
- Email: qhsh9713@gmail.com
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNet(nn.Module):
    """Basic CNN model definition."""

    def __init__(self) -> None:
        """Init the CNN model."""
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Model feed forward function."""
        tensor = F.relu(self.conv1(tensor))
        tensor = F.relu(self.conv2(tensor))

        tensor = F.max_pool2d(tensor, 2)
        tensor = self.dropout1(tensor)
        tensor = torch.flatten(tensor, 1)

        tensor = F.relu(self.fc1(tensor))
        tensor = self.dropout2(tensor)
        tensor = self.fc2(tensor)
        output = F.log_softmax(tensor, dim=1)
        return output

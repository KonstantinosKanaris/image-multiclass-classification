"""
Contains PyTorch model code to instantiate a TinyVGG model from the CNN
explainer website.
"""

import torch
from torch import nn


class TinyVGG(nn.Module):
    """Creates the TinyVGG architecture.

    Replicates the TinyVGG architecture from the CNN explainer
    website in PyTorch. See the original architecture here:
    https://poloclub.github.io/cnn-explainer/.

    Args:
        input_shape (int):
          An integer indicating number of input channels.
        hidden_units (int):
          An integer indicating number of hidden units between layers.
        output_shape (int):
          An integer indicating number of output units.
    """

    def __init__(
        self,
        output_shape: int,
        input_shape: int = 3,
        hidden_units: int = 16,
    ) -> None:
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=input_shape,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=0,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=0,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=0,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=0,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units * 13 * 13, out_features=output_shape),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(
            self.conv_block_2(self.conv_block_1(x))
        )  # <- leverages the benefits of operator fusion

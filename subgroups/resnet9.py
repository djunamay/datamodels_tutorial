"""
This module defines a ResNet9 model and provides utility functions for constructing the model and loading data.

Classes:
    Mul: A custom PyTorch module that multiplies its input by a given weight.
    Flatten: A custom PyTorch module that flattens its input tensor.
    Residual: A custom PyTorch module that adds its input to the output of a given module.

Functions:
    construct_rn9: Constructs a ResNet9 model.
    get_dataloader: Returns a DataLoader for the CIFAR-10 dataset.

Note:
    The classes `Mul`, `Flatten`, and `Residual`, and the functions `construct_rn9` and `get_dataloader` were adapted from https://github.com/MadryLab/trak
"""

import os
from pathlib import Path
from tqdm.auto import tqdm
import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.nn import CrossEntropyLoss, Conv2d, BatchNorm2d
from torch.optim import SGD, lr_scheduler
import torchvision
import warnings

class Mul(torch.nn.Module):
    """
    A custom PyTorch module that multiplies its input by a given weight.

    Parameters
    ----------
    weight : float
        The weight to multiply the input by.
    """

    def __init__(self, weight: float):
        super(Mul, self).__init__()
        self.weight = weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that multiplies the input by the weight.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        torch.Tensor
            The output tensor after multiplication.
        """
        return x * self.weight


class Flatten(torch.nn.Module):
    """
    A custom PyTorch module that flattens its input tensor.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that flattens the input tensor.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        torch.Tensor
            The flattened output tensor.
        """
        return x.view(x.size(0), -1)


class Residual(torch.nn.Module):
    """
    A custom PyTorch module that adds its input to the output of a given module.

    Parameters
    ----------
    module : torch.nn.Module
        The module whose output will be added to the input.
    """

    def __init__(self, module: torch.nn.Module):
        super(Residual, self).__init__()
        self.module = module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that adds the input to the output of the module.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        torch.Tensor
            The output tensor after addition.
        """
        return x + self.module(x)


def construct_rn9(num_classes: int = 10) -> torch.nn.Module:
    """
    Constructs a ResNet9 model.

    Parameters
    ----------
    num_classes : int, optional
        The number of output classes (default is 10).

    Returns
    -------
    torch.nn.Module
        The constructed ResNet9 model.
    """

    def conv_bn(channels_in: int, channels_out: int, kernel_size: int = 3, stride: int = 1, padding: int = 1, groups: int = 1) -> torch.nn.Sequential:
        """
        Constructs a sequence of convolution, batch normalization, and ReLU layers.

        Parameters
        ----------
        channels_in : int
            The number of input channels.
        channels_out : int
            The number of output channels.
        kernel_size : int, optional
            The size of the convolution kernel (default is 3).
        stride : int, optional
            The stride of the convolution (default is 1).
        padding : int, optional
            The padding of the convolution (default is 1).
        groups : int, optional
            The number of groups for the convolution (default is 1).

        Returns
        -------
        torch.nn.Sequential
            The constructed sequence of layers.
        """
        return torch.nn.Sequential(
            torch.nn.Conv2d(channels_in, channels_out, kernel_size=kernel_size,
                            stride=stride, padding=padding, groups=groups, bias=False),
            torch.nn.BatchNorm2d(channels_out),
            torch.nn.ReLU(inplace=True)
        )

    model = torch.nn.Sequential(
        conv_bn(3, 64, kernel_size=3, stride=1, padding=1),
        conv_bn(64, 128, kernel_size=5, stride=2, padding=2),
        Residual(torch.nn.Sequential(conv_bn(128, 128))),
        conv_bn(128, 256, kernel_size=3, stride=1, padding=1),
        torch.nn.MaxPool2d(2),
        Residual(torch.nn.Sequential(conv_bn(256, 256))),
        conv_bn(256, 128, kernel_size=3, stride=1, padding=0),
        torch.nn.AdaptiveMaxPool2d((1, 1)),
        Flatten(),
        torch.nn.Linear(128, num_classes, bias=False),
        Mul(0.2)
    )
    return model

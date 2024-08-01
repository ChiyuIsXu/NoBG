# -*- coding: utf-8 -*-
"""UNet and NestedUNet implementation"""

import numpy as np

import torch
from torch import nn
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter

import matplotlib.pyplot as plt

# export __all__
# Architecture: UNet, NestedUNet
__all__ = ["VGGBlock"]

# Set random seed for reproducibility
# np.random.seed(42)
# torch.manual_seed(42)


class VGGBlock(nn.Module):
    """A VGG-style convolutional block with two convolutional layers followed by batch normalization and ReLU activation.
    Visual Geometry Group (VGG) is a convolutional neural network architecture that was proposed by K. Simonyan and A. Zisserman in the paper "Very Deep Convolutional Networks for Large-Scale Image Recognition".

    Attributes:
        relu (nn.ReLU): ReLU activation function.
        conv1 (nn.Conv2d): First convolutional layer.
        bn1 (nn.BatchNorm2d): Batch normalization after the first convolutional layer.
        conv2 (nn.Conv2d): Second convolutional layer.
        bn2 (nn.BatchNorm2d): Batch normalization after the second convolutional layer.
    """

    def __init__(self, in_channels, middle_channels, out_channels):
        """
        Initializes the VGGBlock with the given input, middle, and output channels.

        Args:
            in_channels (int): The number of channels in the input.
            middle_channels (int): The number of channels after the first convolution.
            out_channels (int): The number of channels after the second convolution.

        Example usage:
            >>> block = VGGBlock(in_channels=3, middle_channels=64, out_channels=64)
        """
        super(VGGBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        """
        Forward pass through the VGGBlock.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
                - batch_size: Number of images in the batch.
                - in_channels: Number of channels in the input image.
                - height: Height of the input image.
                - width: Width of the input image.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height, width).
                - batch_size: Number of images in the batch.
                - out_channels: Number of channels in the output tensor.
                - height: Height of the output tensor, same as input image height.
                - width: Width of the output tensor, same as input image width.

        Example usage:
            >>> block = VGGBlock(in_channels=3, middle_channels=64, out_channels=64)
            >>> input_tensor = torch.randn(1, 3, 224, 224)
            >>> output = block.forward(input_tensor)
            >>> print(output.shape)  # (1, 64, 224, 224)
        """
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


def main():
    print("\033[1;34m")
    print("================ Architectures ================")
    # Define a 3D sample input with the shape (batch_size, num_channels, height, width)
    input_tensor = torch.randn(1, 3, 256, 256)
    print(f"Input shape: {input_tensor.shape}")

    print("===============================================")
    # Create a VGGBlock instance
    print("VGGBlock:")

    input_channels = 3
    middle_channels = 64
    output_channels = 64
    print("Input channels:", input_channels)
    print("Middle channels:", middle_channels)
    print("Output channels:", output_channels)

    vgg_block = VGGBlock(
        in_channels=input_channels,
        middle_channels=middle_channels,
        out_channels=output_channels,
    )
    output_tensor = vgg_block(input_tensor)
    print(f"Output shape: {output_tensor.shape}")

    print("===============================================")
    print("\033[0m")

    # show_tensor_image(input_tensor)


if __name__ == "__main__":
    main()

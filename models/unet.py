# -*- coding: utf-8 -*-
"""UNet and NestedUNet implementation"""

import numpy as np

import torch
from torch import nn

import matplotlib.pyplot as plt

# export __all__
# Architecture: UNet, NestedUNet
__all__ = ["UNet2D", "NestedUNet2D"]

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


class UNet2D(nn.Module):
    """
    U-Net: Convolutional Networks for Biomedical Image Segmentation.

    This model performs semantic segmentation using a U-Net architecture with
    VGG-like convolutional blocks.

    Attributes:
        pool (nn.MaxPool2d): Max pooling layer.
        up (nn.Upsample): Upsampling layer.
        conv0_0 (VGGBlock): Initial VGGBlock with input channels.
        conv1_0 (VGGBlock): VGGBlock at the first downsampling stage.
        conv2_0 (VGGBlock): VGGBlock at the second downsampling stage.
        conv3_0 (VGGBlock): VGGBlock at the third downsampling stage.
        conv4_0 (VGGBlock): VGGBlock at the fourth downsampling stage.
        conv3_1 (VGGBlock): VGGBlock at the first upsampling stage.
        conv2_2 (VGGBlock): VGGBlock at the second upsampling stage.
        conv1_3 (VGGBlock): VGGBlock at the third upsampling stage.
        conv0_4 (VGGBlock): VGGBlock at the fourth upsampling stage.
        final (nn.Conv2d): Final 1x1 convolution to produce segmentation map.
    """

    def __init__(self, num_classes, x_channels=3):
        """
        Initialize the UNet model.

        Args:
            num_classes (int): Number of output classes for segmentation.
            x_channels (int): Number of input channels. For example, 3 for RGB images and 1 for grayscale images.

        Example usage:
            >>> model = UNet(num_classes=2, x_channels=3)
        """
        super(UNet2D, self).__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.conv0_0 = VGGBlock(x_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv3_1 = VGGBlock(nb_filter[3] + nb_filter[4], nb_filter[3], nb_filter[3])
        self.conv2_2 = VGGBlock(nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv1_3 = VGGBlock(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_4 = VGGBlock(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, x):
        """
        Forward pass of the UNet model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_channels, height, width).
                - batch_size: Number of images in the batch.
                - num_channels: Number of channels in the input image. For RGB images, it is 3; for grayscale images, it is 1.
                - height: Height of the input image.
                - width: Width of the input image.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes, height, width).
                - batch_size: Number of images in the batch.
                - num_classes: Number of segmentation classes. For binary segmentation, it is 2 (foreground and background).
                - height: Height of the output segmentation mask, same as input image height.
                - width: Width of the output segmentation mask, same as input image width.

        Example usage:
            >>> model = UNet(num_classes=2, x_channels=3)
            >>> input_tensor = torch.randn(1, 3, 224, 224)
            >>> output = model(input_tensor)
            >>> print(output.shape)  # (1, 2, 224, 224)
        """
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], dim=1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], dim=1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], dim=1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], dim=1))

        output_nested = self.final(x0_4)
        return output_nested


class NestedUNet2D(nn.Module):
    """
    Nested U-Net: Deep Supervised Image Segmentation Model

    This model performs semantic segmentation using a nested U-Net architecture with
    VGG-like convolutional blocks and optional deep supervision.

    Attributes:
        deep_supervision (bool): Whether to use deep supervision.
        pool (nn.MaxPool2d): Max pooling layer.
        up (nn.Upsample): Upsampling layer.
        conv0_0 to conv0_4, conv1_0 to conv1_3, conv2_0 to conv2_2, conv3_0 to conv3_1, conv4_0 (VGGBlock): VGG-like convolutional blocks.
        final (nn.Conv2d): Final 1x1 convolution for output segmentation map (used when deep_supervision is False).
        final1 to final4 (nn.Conv2d): Final 1x1 convolution for deep supervised outputs (used when deep_supervision is True).
    """

    def __init__(self, num_classes, x_channels=3, deep_supervision=False):
        """
        Initialize the NestedUNet model.

        Args:
            num_classes (int): Number of output classes for segmentation.
            x_channels (int): Number of input channels. For example, 3 for RGB images and 1 for grayscale images.
            deep_supervision (bool): If True, enables deep supervision with multiple output layers.
        """
        super(NestedUNet2D, self).__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.deep_supervision = deep_supervision
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.conv0_0 = VGGBlock(x_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = VGGBlock(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3] + nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(
            nb_filter[0] * 2 + nb_filter[1], nb_filter[0], nb_filter[0]
        )
        self.conv1_2 = VGGBlock(
            nb_filter[1] * 2 + nb_filter[2], nb_filter[1], nb_filter[1]
        )
        self.conv2_2 = VGGBlock(
            nb_filter[2] * 2 + nb_filter[3], nb_filter[2], nb_filter[2]
        )

        self.conv0_3 = VGGBlock(
            nb_filter[0] * 3 + nb_filter[1], nb_filter[0], nb_filter[0]
        )
        self.conv1_3 = VGGBlock(
            nb_filter[1] * 3 + nb_filter[2], nb_filter[1], nb_filter[1]
        )

        self.conv0_4 = VGGBlock(
            nb_filter[0] * 4 + nb_filter[1], nb_filter[0], nb_filter[0]
        )

        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, x):
        """
        Forward pass of the NestedUNet model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_channels, height, width).
                              - batch_size: Number of images in the batch.
                              - num_channels: Number of channels in the input image.
                              - height: Height of the input image.
                              - width: Width of the input image.

        Returns:
            torch.Tensor or List[torch.Tensor]:
                If deep_supervision is False:
                    - Output tensor of shape (batch_size, num_classes, height, width).
                      - batch_size: Number of images in the batch.
                      - num_classes: Number of segmentation classes.
                      - height: Height of the output segmentation mask, same as input image height.
                      - width: Width of the output segmentation mask, same as input image width.

                If deep_supervision is True:
                    - List of 4 output tensors, each of shape (batch_size, num_classes, height, width).
                      - Each tensor corresponds to the output from different depths of the network for deep supervision.
                      - batch_size: Number of images in the batch.
                      - num_classes: Number of segmentation classes.
                      - height: Height of the output segmentation mask, same as input image height.
                      - width: Width of the output segmentation mask, same as input image width.
        """
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], dim=1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], dim=1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], dim=1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], dim=1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], dim=1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], dim=1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], dim=1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], dim=1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], dim=1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], dim=1))

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]
        else:
            output_nested = self.final(x0_4)
            return output_nested


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
    # Instantiate the UNet model and output it
    print("UNet:")
    unet = UNet2D(num_classes=6)
    output_tensor = unet(input_tensor)

    print(f"Output shape: {output_tensor.shape}")

    print("===============================================")
    # Instantiate the NestedUNet model and output
    print("NestedUNet:")
    nested_unet = NestedUNet2D(num_classes=6, x_channels=3, deep_supervision=True)
    outputs = nested_unet(input_tensor)

    if isinstance(outputs, list):
        for i, output in enumerate(outputs):
            print(f"Output {i+1} shape: {output.shape}")
    else:
        print(f"Output shape: {outputs.shape}")

    print("===============================================")
    print("\033[0m")

    # show_tensor_image(input_tensor)


if __name__ == "__main__":
    main()

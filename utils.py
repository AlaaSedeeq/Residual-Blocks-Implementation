import torch 
import torch.nn as nn
import torch.nn .functional as F


def conv3x3(in_channels: int,
            out_channels: int,
            padding: int = 1,
            stride: int = 1):
    """
    3x3 convolution with padding for Basic Block
    """
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3, # 3x3
        stride=stride,
        padding=padding,
        bias=False,
    )


def conv1x1(in_channels: int,
            out_channels: int,
            padding: int = 1,
            stride: int = 1):
    """
    1x1 convolution (channel-wise pooling) for Bottleneck Block 
    structure and the conv-connection in both Bottleneck and 
    Basic blocks residual connetions.
    out = ((i+2p-K)/S)+1, here it will have the same dimensions
    of input but down sample the depth or number of feature maps.
    It's like a linear weighting or projection of the input.
    """
    return nn.Conv2d(
        in_channels,
        out_channels, # number of kernels
        padding=padding,
        kernel_size=1, # 1x1
        stride=stride, 
        bias=False
    )

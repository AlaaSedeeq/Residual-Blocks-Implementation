import torch 
import torch.nn as nn
import torch.nn .functional as F
from utils import conv3x3, conv1x1


class BottleneckBlock(nn.Module):
    
    def __init__(
        self, 
        channel_size, 
        shortcut: str = "identity",
        stride: int = 1
    )-> None:
        
        super(BottleneckBlock, self).__init__()
        
        # first layer
        self.conv1 = conv1x1(in_channel_size, channel_size, stride=stride, padding=1)
        # BatchNorm is an element-wise operation and therefore, it does not change the size of our volume.
        self.BN1 = nn.BatchNorm2d(channel_size) 
        
        # second layer
        self.conv2 = conv3x3(channel_size, channel_size, padding=1)
        # BatchNorm is an element-wise operation and therefore, it does not change the size of our volume.
        self.BN2 = nn.BatchNorm2d(channel_size)
        
        # third layer
        self.conv3 = conv1x1(channel_size, channel_size, padding=1)
        # BatchNorm is an element-wise operation and therefore, it does not change the size of our volume.
        self.BN3 = nn.BatchNorm2d(channel_size)
        
        # for projection short-cut (same input(x) channels, and downsampling it to channel_size)
        self.shortcut = shortcut 
        self.shortcut = conv1x1(in_channel_size, channel_size, stride=stride)
        # activation function
        self.relu = nn.ReLU(inplace=True)
        
    def forward(
        self, 
        x:Tensor
    ) -> Tensor:
        # shortcut path may be identity short-cut or projection short-cut
        # identity short-cut
        if self.shortcut.lower()=='identity':
            shortcut = x
            
        elif self.shortcut.lower()=='projection':
            # projection short-cut
            shortcut = self.shortcut(x)
        
        else:
            raise ValueError("ResidualBlock only supports identity or projection shortcut")
            
        # forward path
        # first layer
        out = self.conv1(x)
        out = self.BN(out)
        out = self.relu(out)
        # second layer
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        # second layer
        out = self.conv3(out)
        out = self.bn3(out)
        #out (layers output + shortcut output)
        out += shortcut
        out = self.relu(out)
        return out
import torch 
import torch.nn as nn
import torch.nn .functional as F
from importlib.machinery import SourceFileLoader

utils = SourceFileLoader("utils", "./utils/utils.py").load_module()


class BasicBlock(nn.Module):
    
    def __init__(
        self, 
        channel_size,
        shortcut_method: str = "identity",
        stride: int = 1
    )-> None:
        
        super(BasicBlock, self).__init__()
        # first layer
        self.conv1 = utils.conv3x3(channel_size, channel_size, stride=stride, padding=1)
        # BatchNorm is an element-wise operation and therefore, it does not change the size of our volume.
        self.BN1 = nn.BatchNorm2d(channel_size) 
        
        # second layer
        self.conv2 = utils.conv3x3(channel_size, channel_size, padding=1)
        # BatchNorm is an element-wise operation and therefore, it does not change the size of our volume.
        self.BN2 = nn.BatchNorm2d(channel_size)
        
        # for projection short-cut (same input(x) channels, and downsampling it to channel_size)
        self.shortcut_method = shortcut_method 
        self.shortcut = utils.conv1x1(channel_size, channel_size, stride=stride)
        # activation function
        self.relu = nn.ReLU(inplace=True)
    
    def forward(
        self, 
        x:torch.Tensor
    ) -> torch.Tensor:
        # shortcut path may be identity short-cut OR projection short-cut
        # identity short-cut
        if self.shortcut_method.lower()=='identity':
            shortcut = x
        elif self.shortcut_method.lower()=='projection':
            # projection short-cut
            shortcut = self.shortcut(x)
        else:
            raise ValueError("ResidualBlock only supports identity or projection shortcut")
            
        ################
        # Forward Path #
        ################
        # first layer
        out = self.conv1(x)
        out = self.BN1(out)
        out = self.relu(out)
        # second layer
        out = self.conv2(out)
        out = self.BN2(out)
        
        ##########
        # Output #
        ##########
        #out (layers output + shortcut output)
        out += shortcut
        out = self.relu(out)
        return out

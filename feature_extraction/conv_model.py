import math
import torch
import torch.nn as nn
from torch.nn import Conv1d, Conv2d, Conv3d
from torch.nn import ConvTranspose1d, ConvTranspose2d, ConvTranspose3d
import torch.nn.functional as F

from utils.utils import ShiftedReLU


class ConvLayer(nn.Module):
    """
    An implementation of a Convolutional Sparse Layer
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=0, convo_dim=2):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_dim = convo_dim

        if isinstance(kernel_size, int):
            self.kernel_size = self.conv_dim * (kernel_size,)
        else:
            self.kernel_size = kernel_size

        if isinstance(stride, int):
            self.stride = self.conv_dim * (stride,)
        else:
            self.stride = stride

        if isinstance(padding, int):
            self.padding = self.conv_dim * (padding,)
        else:
            self.padding = padding

        if self.conv_dim == 1:
            self.convo = Conv1d(in_channels, out_channels, kernel_size, stride, padding)
            self.deconvo = ConvTranspose1d(out_channels, in_channels, kernel_size, stride, padding)
        elif self.conv_dim == 2:
            self.convo = Conv2d(in_channels, out_channels, kernel_size, stride, padding)
            self.deconvo = ConvTranspose2d(out_channels, in_channels, kernel_size, stride, padding)
        elif self.conv_dim == 3:
            self.convo = Conv3d(in_channels, out_channels, kernel_size, stride, padding)
            self.deconvo = ConvTranspose3d(out_channels, in_channels, kernel_size, stride, padding)
        else:
            raise ValueError("Conv_dim must be 1, 2, or 3")

    def loss(self, images, recon):
        loss = 0.5 * (1/images.shape[0]) * torch.sum(
            torch.pow(images - recon, 2))
        return loss
    
    def get_activations(self, images):
        return F.relu(self.convo(images))

    def forward(self, images):
        batch_size = images.size(0)
        
        x = F.relu(self.convo(images))

        x = F.relu(self.deconvo(x))

        return x
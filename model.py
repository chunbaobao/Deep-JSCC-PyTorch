import torch
import torch.nn as nn


class _ConvWithPReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0):
        super(_ConvWithPReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.prelu(x)
        return x


class _TransConvWithPReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, activate=nn.PReLU(), padding=0):
        super(_TransConvWithPReLU, self).__init__()
        self.transconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
        self.activate = activate

    def forward(self, x):
        x = self.transconv(x)
        x = self.activate(x)
        return x


class Nomalization(nn.Module):
    def __init__(self, in_channels):
        super(Nomalization, self).__init__()
        self.norm = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        x = self.norm(x)
        return x

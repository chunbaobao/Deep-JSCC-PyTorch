# -*- coding: utf-8 -*-
"""
Created on Tue Dec  11:00:00 2023

@author: chun
"""

import torch
import torch.nn as nn
import numpy as np


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


def _image_normalization(tensor, norm_type):
    if norm_type == 'nomalization':
        return tensor / 255.0
    elif norm_type == 'denormalization':
        return tensor * 255.0
    else:
        raise Exception('Unknown type of normalization')


def _NormlizationLayer(norm_type='nomalization'):
    pass


def ratio2filter_size(x, ratio):
    before_size = np.prod(x.size())
    after_size = before_size*ratio
    encoder_temp = Encoder(c=after_size)


class Encoder(nn.Module):
    def __init__(self, c, is_temp=False):
        super(Encoder, self).__init__()
        self.is_temp = is_temp
        self.imgae_normalization = _image_normalization(norm_type='nomalization')
        self.conv1 = _ConvWithPReLU(in_channels=3, out_channels=16, kernel_size=5, stride=2)
        self.conv2 = _ConvWithPReLU(in_channels=16, out_channels=32, kernel_size=5, stride=2)
        self.conv3 = _ConvWithPReLU(in_channels=32, out_channels=32, kernel_size=5, stride=1)
        self.conv4 = _ConvWithPReLU(in_channels=32, out_channels=32, kernel_size=5, stride=1)
        self.conv5 = _ConvWithPReLU(in_channels=32, out_channels=c, kernel_size=5, stride=1)
        self.norm = _NormlizationLayer(norm_type='nomalization')

    def forward(self, x):
        x = self.imgae_normalization(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        if not self.is_temp:
            x = self.conv5(x)
        z = self.norm(x)
        del x
        return z


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

import torch
import torch.nn as nn
import torch.nn.functional as F


def image_normalization(norm_type):
    def _inner(tensor: torch.Tensor):
        if norm_type == 'nomalization':
            return tensor / 255.0
        elif norm_type == 'denormalization':
            return tensor * 255.0
        else:
            raise Exception('Unknown type of normalization')
    return _inner


def get_psnr(image, gt, max=255):

    mse = F.mse_loss(image, gt)

    psnr = 10 * torch.log10(max**2 / mse)
    return psnr


a = torch.randn(2, 3, 32, 32)
b = image_normalization('nomalization')(a)

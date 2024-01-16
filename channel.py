import torch
import torch.nn as nn

def channel(channel_type='AWGN', snr=20):
    def AWGN_channel(z_hat: torch.Tensor):
        if z_hat.dim() == 4:
            # k = np.prod(z_hat.size()[1:])
            k = torch.prod(torch.tensor(z_hat.size()[1:]))
            sig_pwr = torch.sum(torch.abs(z_hat).square(), dim=(1, 2, 3), keepdim=True)/k
        elif z_hat.dim() == 3:
            # k = np.prod(z_hat.size())
            k = torch.prod(torch.tensor(z_hat.size()))
            sig_pwr = torch.sum(torch.abs(z_hat).square())/k
        noi_pwr = sig_pwr / ( 10 ** (snr / 10))
        noise = torch.randn_like(z_hat) * torch.sqrt(noi_pwr)
        return z_hat + noise

    def Rayleigh_channel(z_hat: torch.Tensor):
        pass

    if channel_type == 'AWGN':
        return AWGN_channel
    elif channel_type == 'Rayleigh':
        return Rayleigh_channel
    else:
        raise Exception('Unknown type of channel')

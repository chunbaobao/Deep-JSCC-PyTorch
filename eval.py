# to be implemented
import torch
import torch.nn as nn
from PIL import Image


def config_parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--channel', default='AWGN', type=str, help='channel type')
    parser.add_argument('--saved', default='./saved', type=str, help='saved_path')
    parser.add_argument('--snr_list', default=range(1, 19, 3), type=list, help='snr_list')
    parser.add_argument('--demo_image', default='./demo/kodim08.png', type=str, help='demo_image')
    return parser.parse_args()

# -*- coding: utf-8 -*-
"""
Created on Tue Dec  11:00:00 2023

@author: chun
"""

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import tqdm
from model import DeepJSCC


def config_parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=2048, type=int, help='Random seed')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('optimizer', default='Adam', type=str, help='optimizer')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay')
    parser.add_argument('--channel', default='AWGN', type=str, help='weight decay')
    parser.add_argument('--saved', default='./saved', type=str, help='saved_path')
    return parser.parse_args()


def main():
    args = config_parser()

    # load data
    transform = transforms.Compose([transforms.ToTensor(), ])
    train_dataset = datasets.CIFAR10(root='./Dataset/cifar-10-batches-py/', train=True,
                                     download=True, transform=transform)

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
    test_dataset = datasets.MNIST(root='./Dataset/cifar-10-batches-py/', train=False,
                                  download=True, transform=transform)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size)


def train():
    pass


if __name__ == '__main__':
    main()

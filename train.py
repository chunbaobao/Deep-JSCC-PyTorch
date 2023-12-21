# -*- coding: utf-8 -*-
"""
Created on Tue Dec  17:00:00 2023

@author: chun
"""

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.optim as optim
import tqdm
from model import DeepJSCC, ratio2filtersize
from torch.nn.parallel import DataParallel
from channel import channel


def config_parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=2048, type=int, help='Random seed')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--epochs', default=100, type=int, help='number of epochs')
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight_decay', default=1e-3, type=float, help='weight decay')
    parser.add_argument('--channel', default='AWGN', type=str, help='channel type')
    parser.add_argument('--saved', default='./saved', type=str, help='saved_path')
    parser.add_argument('--snr_list', default=range(1, 19, 3), type=list, help='snr_list')
    parser.add_argument('--ratio_list', default=[1/6, 1/12], type=list, help='ratio_list')
    parser.add_argument('--early_stop', default=True, type=bool, help='early_stop')
    return parser.parse_args()


def main():
    args = config_parser()

    print("Training Start")
    # for ratio in args.ratio_list:
    #     for snr in args.snr_list:
    #         train(args, ratio, snr)
    train(args, 1/6, 20)


def train(args: config_parser(), ratio: float, snr: float):

    print("training with ratio: {}, snr: {}, channel: {}".format(ratio, snr, args.channel))

    # load data
    transform = transforms.Compose([transforms.ToTensor(), ])
    train_dataset = datasets.CIFAR10(root='./Dataset/', train=True,
                                     download=True, transform=transform)

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
    test_dataset = datasets.MNIST(root='./Dataset/', train=False,
                                  download=True, transform=transform)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size)
    image_fisrt = train_dataset.__getitem__(0)[0]
    c = ratio2filtersize(image_fisrt, ratio)
    model = DeepJSCC(c=c, channel_type=args.channel, snr=snr)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    epoch_loop = tqdm((args.epochs), total=len(args.epochs), leave=False)
    for epoch in epoch_loop:
        run_loss = 0.0
        for images, _ in tqdm((train_loader), leave=False):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, images)
            loss.backward()
            optimizer.step()
            run_loss += loss.item()
        epoch_loop.set_description(f'Epoch [{epoch}/{args.epochs}]')
        epoch_loop.set_postfix(loss=run_loss)
    save_model(model, args.saved + '/model_{}_{}.pth'.format(ratio, snr))


def save_model(model, path):
    torch.save(model.state_dict(), path)


if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
"""
Created on Tue Dec  17:00:00 2023

@author: chun
"""
import os
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader, RandomSampler
import torch.optim as optim
from tqdm import tqdm
from model import DeepJSCC, ratio2filtersize
from torch.nn.parallel import DataParallel


def config_parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=2048, type=int, help='Random seed')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--epochs', default=100, type=int, help='number of epochs')
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay')
    parser.add_argument('--channel', default='AWGN', type=str, help='channel type')
    parser.add_argument('--saved', default='./saved', type=str, help='saved_path')
    parser.add_argument('--snr_list', default=range(1, 19, 3), type=list, help='snr_list')
    parser.add_argument('--ratio_list', default=[1/6, 1/12], type=list, help='ratio_list')
    parser.add_argument('--num_workers', default=0, type=int, help='num_workers')
    return parser.parse_args()


def main():
    args = config_parser()

    print("Training Start")
    # for ratio in args.ratio_list:
    #     for snr in args.snr_list:
    #         train(args, ratio, snr)
    train(args, 1/6, 20)


def train(args: config_parser(), ratio: float, snr: float):

    device = torch.device('cuda:1')
    # load data
    transform = transforms.Compose([transforms.ToTensor(), ])
    train_dataset = datasets.CIFAR10(root='./Dataset/', train=True,
                                     download=True, transform=transform)

    train_loader = DataLoader(train_dataset, shuffle=True,
                              batch_size=args.batch_size, num_workers=args.num_workers)
    test_dataset = datasets.CIFAR10(root='./Dataset/', train=False,
                                    download=True, transform=transform)
    test_loader = RandomSampler(test_dataset, replacement=True, num_samples=args.batch_size)

    print("training with ratio: {}, snr_db: {}, channel: {}".format(ratio, snr, args.channel))

    image_fisrt = train_dataset.__getitem__(0)[0]
    c = ratio2filtersize(image_fisrt, ratio)
    model = DeepJSCC(c=c, channel_type=args.channel, snr=snr).cuda(device=device)

    criterion = nn.MSELoss().cuda(device=device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    epoch_loop = tqdm(range(args.epochs), total=args.epochs, leave=False)

    for epoch in epoch_loop:
        run_loss = 0.0
        for images, _ in tqdm((train_loader), leave=False):
            optimizer.zero_grad()
            images = images.cuda(device=device)
            outputs = model(images)
            loss = criterion(outputs, images)
            loss.backward()
            optimizer.step()
            run_loss += loss.item()

        epoch_loop.set_description(f'Epoch [{epoch}/{args.epochs}]')
        epoch_loop.set_postfix(loss=run_loss/len(train_loader))
    save_model(model, args.saved + '/model_{:2f}_{:2f}.pth'.format(ratio, snr))


def save_model(model, path):
    os.makedirs(path, exist_ok=True)
    torch.save(model.state_dict(), path)
    print("Model saved in {}".format(path))


if __name__ == '__main__':
    main()

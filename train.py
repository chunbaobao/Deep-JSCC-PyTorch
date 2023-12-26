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
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
from model import DeepJSCC, ratio2filtersize
from torch.nn.parallel import DataParallel
from utils import image_normalization
from fractions import Fraction
from dataset import Vanilla


def config_parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=2048, type=int, help='Random seed')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--epochs', default=256, type=int, help='number of epochs')
    parser.add_argument('--batch_size', default=256, type=int, help='batch size')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay')
    parser.add_argument('--channel', default='AWGN', type=str,
                        choices=['AWGN', 'Rayleigh'], help='channel type')
    parser.add_argument('--saved', default='./saved', type=str, help='saved_path')
    parser.add_argument('--snr_list', default=['19', '13',
                        '7', '4', '1'], nargs='+', help='snr_list')
    parser.add_argument('--ratio_list', default=['1/3',
                        '1/6', '1/12'], nargs='+', help='ratio_list')
    parser.add_argument('--num_workers', default=0, type=int, help='num_workers')
    parser.add_argument('--dataset', default='cifar10', type=str,
                        choices=['cifar10', 'imagenet'], help='dataset')
    parser.add_argument('--parallel', default=False, type=bool, help='parallel')
    parser.add_argument('--if_scheduler', default=False, type=bool, help='if_scheduler')
    parser.add_argument('--step_size', default=640, type=int, help='scheduler')
    parser.add_argument('--device', default='cuda:0', type=str, help='device')
    return parser.parse_args()


def main():
    args = config_parser()
    args.snr_list = list(map(float, args.snr_list))
    args.ratio_list = list(map(lambda x: float(Fraction(x)), args.ratio_list))
    print("Training Start")
    for ratio in args.ratio_list:
        for snr in args.snr_list:
            train(args, ratio, snr)

def train(args: config_parser(), ratio: float, snr: float):

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    # load data
    if args.dataset == 'cifar10':
        transform = transforms.Compose([transforms.ToTensor(), ])
        train_dataset = datasets.CIFAR10(root='./Dataset/', train=True,
                                         download=True, transform=transform)

        train_loader = DataLoader(train_dataset, shuffle=True,
                                  batch_size=args.batch_size, num_workers=args.num_workers)
        test_dataset = datasets.CIFAR10(root='./Dataset/', train=False,
                                        download=True, transform=transform)
        test_loader = DataLoader(test_dataset, shuffle=True,
                                 batch_size=args.batch_size, num_workers=args.num_workers)
    elif args.dataset == 'imagenet':
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Resize((128, 128))])  # the size of paper is 128
        print("loading data of imagenet")
        train_dataset = datasets.ImageFolder(root='./Dataset/ImageNet/train', transform=transform)

        train_loader = DataLoader(train_dataset, shuffle=True,
                                  batch_size=args.batch_size, num_workers=args.num_workers)
        test_dataset = Vanilla(root='./Dataset/ImageNet/val', transform=transform)
        test_loader = DataLoader(test_dataset, shuffle=True,
                                 batch_size=args.batch_size, num_workers=args.num_workers)
    else:
        raise Exception('Unknown dataset')

    print("training with ratio: {:.2f}, snr_db: {}, channel: {}".format(ratio, snr, args.channel))

    image_fisrt = train_dataset.__getitem__(0)[0]
    c = ratio2filtersize(image_fisrt, ratio)
    print("the inner channel is {}".format(c))
    model = DeepJSCC(c=c, channel_type=args.channel, snr=snr)

    if args.parallel and torch.cuda.device_count() > 1:
        model = DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
        model = model.cuda()
        criterion = nn.MSELoss(reduction='mean').cuda()
    else:
        model = model.to(device)
        criterion = nn.MSELoss(reduction='mean').to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.if_scheduler:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.5)

    epoch_loop = tqdm(range(args.epochs), total=args.epochs, leave=True)
    for epoch in epoch_loop:
        run_loss = 0.0
        for images, _ in tqdm((train_loader), leave=False):
            optimizer.zero_grad()
            images = images.cuda() if args.parallel else images.to(device)
            outputs = model(images)
            outputs = image_normalization('denormalization')(outputs)
            images = image_normalization('denormalization')(images)
            loss = criterion(outputs, images)
            loss.backward()
            optimizer.step()
            run_loss += loss.item()
        if args.if_scheduler: # the scheduler is wrong before
            scheduler.step()
        with torch.no_grad():
            model.eval()
            test_mse = 0.0
            for images, _ in tqdm((test_loader), leave=False):
                images = images if args.parallel else images.to(device)
                outputs = model(images)
                images = image_normalization('denormalization')(images)
                outputs = image_normalization('denormalization')(outputs)
                loss = criterion(outputs, images)
                test_mse += loss.item()
            model.train()
        epoch_loop.set_postfix(loss=run_loss/len(train_loader), test_mse=test_mse/len(test_loader))
        print("epoch: {}, loss: {:.4f}, test_mse: {:.4f} lr:{}".format(
            epoch, run_loss/len(train_loader), test_mse/len(test_loader), optimizer.param_groups[0]['lr']))
    save_model(model, args.saved, args.saved +
               '/{}_{}_{:.2f}_{:.2f}_{}.pth'.format(args.dataset, args.epochs, ratio, snr, c))


def save_model(model, dir, path):
    os.makedirs(dir, exist_ok=True)
    torch.save(model.state_dict(), path)
    print("Model saved in {}".format(path))


if __name__ == '__main__':
    main()

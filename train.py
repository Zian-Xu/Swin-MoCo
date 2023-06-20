import argparse
import copy
import math
import os
import random
import sys
from functools import partial

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.transforms.functional as f
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import swin_transformer
import utils
from swin_moco import SwinMoCo


def preparation(args):
    if args.seed is not None:
        print(f'Using seed {args.seed} for deterministic training.')
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
    else:
        cudnn.benchmark = True

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))


def load_weight(model, args):
    if args.weights is None:
        print('None weights file path.')
    elif not os.path.exists(args.weights):
        print(f'Weights file: "{args.weights}" not exist.')
    else:
        weights_dict = torch.load(args.weights, map_location='cuda:' + str(args.gpu))["model"]
        model_dict = model.state_dict()
        full_dict = copy.deepcopy(weights_dict)

        for key in weights_dict:
            full_dict['base_encoder.' + key] = full_dict[key]
            full_dict['momentum_encoder.' + key] = full_dict[key]
            del full_dict[key]

        for k in list(full_dict.keys()):
            if k in model_dict:
                if full_dict[k].shape != model_dict[k].shape:
                    print(f"Delete: '{k}'; Weight shape: {full_dict[k].shape}; Model shape: {model_dict[k].shape}")
                    del full_dict[k]
            else:
                del full_dict[k]

        result = model.load_state_dict(full_dict, strict=False)
        print(result)
    return model


def creat_model(args):
    model = SwinMoCo(
        partial(swin_transformer.__dict__['SwinTransformer'],
                patch_size=4,
                in_chans=args.in_chans,
                num_classes=args.num_classes,
                embed_dim=96,
                depths=(2, 2, 2, 2),
                num_heads=(3, 6, 12, 24),
                window_size=7,
                mlp_ratio=4,
                qkv_bias=True,
                drop_rate=0,
                attn_drop_rate=0,
                drop_path_rate=0.1),
        args.moco_dim, args.moco_mlp_dim, args.moco_t)
    torch.cuda.set_device(args.gpu)
    model = model.cuda(args.gpu)
    model = load_weight(model=model, args=args)
    return model


def param_config(args):
    args.optimizer = torch.optim.AdamW(args.model.parameters(), args.lr,
                                       weight_decay=args.weight_decay)
    args.lr = args.lr * args.batch_size / 256
    args.scaler = torch.cuda.amp.GradScaler()
    args.summary_writer = SummaryWriter()
    return args


def load_data(args):
    augmentation_1 = [
        transforms.RandomRotation(degrees=10, interpolation=f.InterpolationMode.BILINEAR),
        transforms.RandomResizedCrop(224, scale=(args.crop_min, 1.)),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([utils.GaussianBlur([.1, 2.])], p=1.0),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()]
    augmentation_2 = [
        transforms.RandomRotation(degrees=10, interpolation=f.InterpolationMode.BILINEAR),
        transforms.RandomResizedCrop(224, scale=(args.crop_min, 1.)),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([utils.GaussianBlur([.1, 2.])], p=0.1),
        transforms.RandomApply([utils.Solarize()], p=0.2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()]

    train_dataset = datasets.ImageFolder(
        args.dataset_path,
        utils.TwoCropsTransform(
            transforms.Compose(augmentation_1),
            transforms.Compose(augmentation_2)))
    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.num_workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    return train_loader


class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def adjust_learning_rate(optimizer, epoch, args):
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs
    else:
        lr = args.lr * 0.5 * (
                1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def adjust_moco_momentum(epoch, args):
    m = 1. - 0.5 * (1. + math.cos(math.pi * epoch / args.epochs)) * (1. - args.moco_m)
    return m


def train(train_loader, model, optimizer, scaler, epoch, args):
    losses = AverageMeter('Loss', ':.4e')
    learning_rates = AverageMeter('LR', ':.4e')
    model.train()
    iters_per_epoch = len(train_loader)
    moco_m = args.moco_m
    train_loader = tqdm(train_loader, file=sys.stdout)

    for i, (images, _) in enumerate(train_loader):
        lr = adjust_learning_rate(optimizer, epoch + i / iters_per_epoch, args)
        learning_rates.update(lr)
        if args.moco_m_cos:
            moco_m = adjust_moco_momentum(epoch + i / iters_per_epoch, args)
        images[0] = images[0].cuda(args.gpu, non_blocking=True)
        images[1] = images[1].cuda(args.gpu, non_blocking=True)

        with torch.cuda.amp.autocast(True):
            loss = model(images[0], images[1], moco_m)
        losses.update(loss.item(), images[0].size(0))
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_loader.desc = "[Epoch {}/{}] loss: {:.3f}, lr: {:.3e}".format(
            epoch + 1, args.epochs, losses.avg, learning_rates.val)
    return losses.avg, learning_rates.avg


def main(args):
    preparation(args=args)
    args.model = creat_model(args=args)
    args = param_config(args=args)
    train_loader = load_data(args=args)

    for epoch in range(args.epochs):
        loss, lr = train(train_loader, args.model, args.optimizer, args.scaler, epoch, args)
        tags = ["loss", "learning_rate"]
        args.summary_writer.add_scalar(tags[0], loss, epoch)
        args.summary_writer.add_scalar(tags[1], lr, epoch)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    torch.save(args.model.state_dict(), os.path.join(args.save_path, 'model.pth'))
    args.summary_writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Swin MoCo Pre-Training')
    parser.add_argument('--dataset-path', default='<Your dataset path>')
    parser.add_argument('--weights', type=str, default='<Your weight file path>')
    parser.add_argument('--save-path', default='model')

    parser.add_argument('--batch-size', default=80, type=int)
    parser.add_argument('--num-workers', default=8, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--warmup-epochs', default=30, type=int)
    parser.add_argument('--lr', default=1e-3, type=float, help='max learning rate')
    parser.add_argument('--seed', default=3407, type=int, help='seed for initializing training')
    parser.add_argument('--crop-min', default=0.08, type=float)
    parser.add_argument('--weight-decay', default=1e-6, type=float)
    parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')

    parser.add_argument('--moco-dim', default=256, type=int,
                        help='feature dimension (default: 256)')
    parser.add_argument('--moco-mlp-dim', default=4096, type=int,
                        help='hidden dimension in MLPs (default: 4096)')
    parser.add_argument('--moco-m', default=0.99, type=float,
                        help='moco momentum of updating momentum encoder (default: 0.99)')
    parser.add_argument('--moco-m-cos', default=True,
                        help='gradually increase moco momentum to 1 with a '
                             'half-cycle cosine schedule')
    parser.add_argument('--moco-t', default=1.0, type=float,
                        help='softmax temperature (default: 1.0)')
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--in_chans', type=int, default=3)

    main(parser.parse_args())

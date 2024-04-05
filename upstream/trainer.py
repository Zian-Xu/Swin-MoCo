import math
import sys

import torch
from tqdm import tqdm


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

    for i, images in enumerate(train_loader):
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
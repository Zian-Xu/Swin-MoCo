import os
import argparse
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
import torch.backends.cudnn

from dataset import MyDataset, collate_fn
from model import SwinUnet
from utils import load_weight, DiceLoss
from trainer import train_one_epoch


def worker_init_fn(worker_id):
    random.seed(1234 + worker_id)


def main(args):
    # Deterministic training
    if args.seed:
        print(f'Use deterministic training with seed {args.seed}.')
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
    else:
        print('Deterministic training is not used.')
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    # Dataset
    train_set = MyDataset(dataset_path=args.dataset_path,
                          output_size=args.output_size)
    train_loader = DataLoader(dataset=train_set,
                              batch_size=args.batch_size,
                              shuffle=True,
                              pin_memory=True,
                              persistent_workers=True,
                              num_workers=8,
                              collate_fn=collate_fn,
                              worker_init_fn=worker_init_fn)

    # Model
    model = SwinUnet(patch_size=4,
                     in_chans=3,
                     num_classes=args.num_classes,
                     embed_dim=96,
                     depths=(2, 2, 2, 2),
                     num_heads=(3, 6, 12, 24),
                     window_size=7,
                     mlp_ratio=4,
                     qkv_bias=True,
                     drop_rate=0,
                     attn_drop_rate=0,
                     drop_path_rate=0.2).to('cuda')
    model = load_weight(model=model, device='cuda', path=args.weights_path)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    loss_func = [CrossEntropyLoss(),
                 DiceLoss(args.num_classes)]

    # Training
    tb_writer = SummaryWriter()
    iter_num = 0
    iter_max = args.epochs * len(train_loader)
    for epoch in range(args.epochs):
        iter_num = train_one_epoch(data_loader=train_loader,
                                   model=model,
                                   optimizer=optimizer,
                                   loss_func=loss_func,
                                   base_lr=args.lr,
                                   tb_writer=tb_writer,
                                   epoch=epoch,
                                   iter_num=iter_num,
                                   iter_max=iter_max)
        # Save model
        if epoch == args.epochs - 1:
            if not os.path.exists(args.save_path):
                os.makedirs(args.save_path)
            torch.save(model.state_dict(), args.save_path + f'/model-{epoch + 1}.pth')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Dataset parameter
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--output_size', type=int, default=224)
    parser.add_argument('--num_classes', type=int, default=9)
    parser.add_argument('--classes_name', type=tuple, default=(
        'background', 'aorta', 'gallbladder', 'left_kidney', 'right_kidney',
        'liver', 'pancreas', 'spleen', 'stomach'))

    # Hyper parameter
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--seed', type=int, default=1234)  # The seed in worker_init_fn needs to be changed manually.

    # Path parameter
    parser.add_argument('--dataset_path', type=str, default=r'C:\文件\数据集\Synapse\train_npz')
    parser.add_argument('--weights_path', type=str, default='../upstream/model/model.pth')
    parser.add_argument('--save_path', type=str, default='trained_model')

    opt = parser.parse_args()
    main(opt)

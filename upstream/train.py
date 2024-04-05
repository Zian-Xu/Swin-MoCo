import os
import copy
import random
import argparse
from functools import partial

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from dataset import MyDataset, collate_fn
import swin_transformer
from swin_moco import SwinMoCo
from trainer import train


def worker_init_fn(worker_id):
    random.seed(1234 + worker_id)


def preparation(args):
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

    if args.gpu is not None:
        print('Use GPU: {} for training'.format(args.gpu))


def load_weight(model, args):
    if args.weights is None:
        print('None weights file path.')
    elif not os.path.exists(args.weights):
        print(f'Weights file: "{args.weights}" not exist.')
    else:
        weights_dict = torch.load(args.weights, map_location='cuda:' + str(args.gpu))['model']
        model_dict = model.state_dict()
        full_dict = copy.deepcopy(weights_dict)

        for key in weights_dict:
            full_dict['base_encoder.' + key] = full_dict[key]
            full_dict['momentum_encoder.' + key] = full_dict[key]
            del full_dict[key]

        for k in list(full_dict.keys()):
            if k in model_dict:
                if full_dict[k].shape != model_dict[k].shape:
                    print(f'Delete: "{k}"; Weight shape: {full_dict[k].shape}; Model shape: {model_dict[k].shape}')
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


def param_config(model, args):
    args.optimizer = torch.optim.AdamW(model.parameters(), args.lr,
                                       weight_decay=args.weight_decay)
    args.lr = args.lr * args.batch_size / 256
    args.scaler = torch.cuda.amp.GradScaler()
    args.summary_writer = SummaryWriter()
    return args


def load_data(args):
    train_dataset = MyDataset(image_path=args.dataset_path,
                              output_size=args.output_size,
                              crop_min=args.crop_min)
    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        drop_last=True,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        persistent_workers=True,
        pin_memory=True,
        collate_fn=collate_fn,
        worker_init_fn=worker_init_fn)
    return train_loader


def main(args):
    preparation(args=args)
    model = creat_model(args=args)
    args = param_config(model, args=args)
    train_loader = load_data(args=args)

    for epoch in range(args.epochs):
        loss, lr = train(train_loader, model, args.optimizer, args.scaler, epoch, args)
        tags = ['loss', 'learning_rate']
        args.summary_writer.add_scalar(tags[0], loss, epoch)
        args.summary_writer.add_scalar(tags[1], lr, epoch)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    torch.save(model.state_dict(), os.path.join(args.save_path, 'model.pth'))
    args.summary_writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Swin MoCo Pre-Training')
    # path param
    parser.add_argument('--dataset-path',
                        default=r'/mnt/dataset/Synapse/train_npz')
    parser.add_argument('--weights', type=str,
                        default=r'../pretrain/swin_tiny_patch4_window7_224.pth')
    parser.add_argument('--save-path', default='model')

    # hyper param
    parser.add_argument('--batch-size', default=80, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--warmup-epochs', default=30, type=int)
    parser.add_argument('--lr', default=1e-3, type=float, help='max learning rate, auto adjust by batch size')
    parser.add_argument('--weight-decay', default=1e-6, type=float)
    parser.add_argument('--output-size', default=224, type=int)
    parser.add_argument('--crop-min', default=0.08, type=float)

    # setting param
    parser.add_argument('--seed', default=1234)  # The seed in worker_init_fn needs to be changed manually.
    parser.add_argument('--num-workers', default=8, type=int)
    parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')

    # model param
    parser.add_argument('--in-chans', type=int, default=3)
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

    main(parser.parse_args())

    # for server
    os.system("export $(cat /proc/1/environ |tr '\\0' '\\n' | grep MATCLOUD_CANCELTOKEN)&&/public/script/matncli node"
              " cancel -url https://matpool.com/api/public/node")

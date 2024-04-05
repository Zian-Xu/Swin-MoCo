# Fixed bug where test preprocessing was different from validation.
# Following Swin-Unet, the evaluation metrics were calculated using h5 files.
import os
import sys
import argparse

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from scipy.ndimage import zoom
from tqdm import tqdm
from medpy import metric
from PIL import Image

from dataset import collate_fn
from model import SwinUnet


class TestDataset(Dataset):
    def __init__(self, test_path: str, output_size: int):
        self.test_path = test_path
        self.test_list = os.listdir(test_path)
        self.output_size = (output_size,) * 2
        self.resize = transforms.Resize(self.output_size)

    def __len__(self):
        return len(self.test_list)

    def __getitem__(self, item):
        data = h5py.File(os.path.join(self.test_path, self.test_list[item]))
        image = data['image'][:]
        label = data['label'][:]

        # Test preprocessing is modified to be the same as validation.
        if image.shape[1:] != self.output_size:
            resize_image = []
            for slice_i in range(image.shape[0]):
                image_slice = Image.fromarray(np.uint8(image[slice_i] * 255))
                resize_image.append(np.asarray(self.resize(image_slice)))
            image = np.float32(resize_image) / 255

        image = torch.from_numpy(image)
        label = torch.from_numpy(label)
        return image, label


def calculate_metric(pred, gt, cal_hd: bool = True):
    # Add format conversion to prevent IDE warnings.
    pred = pred.astype(np.uint8)
    gt = gt.astype(np.uint8)
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        if cal_hd:
            hd95 = metric.binary.hd95(pred, gt)
        else:
            hd95 = np.nan
        return dice, hd95
    # Following Swin-Unet, returns the specified Dice and HD for special cases.
    elif pred.sum() > 0 and gt.sum() == 0:
        return 1, 0  # one case
    else:
        return 0, 0  # Shouldn't happen.


def main(args):
    # dataset
    test_set = TestDataset(test_path=args.test_vol_h5, output_size=224)
    test_loader = DataLoader(dataset=test_set,
                             batch_size=1,
                             shuffle=False,
                             pin_memory=True,
                             persistent_workers=True,
                             num_workers=1,
                             collate_fn=collate_fn)

    # model
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
    msg = model.load_state_dict(torch.load(args.weights_path))
    print(msg)
    model.eval()

    # test
    final_metric = 0
    for index, data in enumerate(test_loader):
        print(f'({index + 1}/{len(test_loader)})')
        pred_list = []
        slice_num = tqdm(range(data[0].shape[1]), file=sys.stdout)
        for slice_i in slice_num:
            image = data[0][:, slice_i].unsqueeze(1).repeat(1, 3, 1, 1).to('cuda')
            with torch.no_grad():
                output = model(image)
                pred = torch.argmax(output, dim=1).squeeze(0)
                pred = pred.cpu().detach().numpy()
                if pred.shape != (args.img_size,) * 2:
                    pred = zoom(pred, (args.img_size / pred.shape[0],
                                       args.img_size / pred.shape[1]), order=0)
                pred_list.append(pred)
                slice_num.desc = '[model]'
        predict = np.asarray(pred_list, dtype=np.uint8)
        label = np.asarray(data[1].squeeze(0))
        classes_metric = []
        classes_num = tqdm(range(1, args.num_classes), file=sys.stdout)
        for class_i in classes_num:
            classes_metric.append(calculate_metric(predict == class_i, label == class_i, cal_hd=args.cal_hd))
            classes_num.desc = '[metric]'
        final_metric = final_metric + np.array(classes_metric)
    final_metric = final_metric / len(test_loader)

    # output
    print('------------------------------')
    if not os.path.exists(os.path.split(args.output_file)[0]):
        os.makedirs(os.path.split(args.output_file)[0])
    with open(args.output_file, 'w') as f:
        for i in range(final_metric.shape[0]):
            info = f'[{args.classes_name[i + 1]}] Dice: {final_metric[i][0] * 100:.2f}, HD: {final_metric[i][1]:.2f}'
            f.write(info + '\n')
            print(info)
        info = f'\n[mean] Dice: {final_metric[:, 0].mean() * 100:.2f}, HD: {final_metric[:, 1].mean():.2f}'
        f.write(info + '\n')
        print(info)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_vol_h5', type=str, default=r'C:\文件\数据集\Synapse\test_vol_h5')
    parser.add_argument('--weights_path', type=str, default='trained_model/model-150.pth')
    parser.add_argument('--output_file', type=str, default='result/metric.txt')
    parser.add_argument('--cal_hd', type=bool, default=True)
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--num_classes', type=int, default=9)
    parser.add_argument('--classes_name', type=tuple, default=(
        'background', 'aorta', 'gallbladder', 'left_kidney', 'right_kidney',
        'liver', 'pancreas', 'spleen', 'stomach'))

    opt = parser.parse_args()
    main(opt)

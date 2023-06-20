## Swin MoCo: Improving Parotid Gland MRI Segmentation Using Contrastive Learning

### Introduction
This is a PyTorch implementation of Swin MoCo.

### Usage
1. Install the required environment in "requirements.txt".
2. Open "train.py" and fill in the dataset path. There should be at least one category folder under this path. The data for training is stored in the category folder.
3. For training Swin MoCo from scratch, you need to set weights=None. For training Swin MoCo with transfer learning, you need to download [Swin Transformer's ImageNet-1K supervised model](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth) and set weights='swin_tiny_patch4_window7_224.pth'.
4. Run "train.py".

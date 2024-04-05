# Swin MoCo: Improving Parotid Gland MRI Segmentation Using Contrastive Learning

## Introduction
This is a PyTorch implementation of Swin MoCo.

We open-sourced this code to help readers reproduce the transfer learning performance of Swin MoCo on the Synapse dataset. 

In comparison to the official code of Swin-Unet, we made the following modifications:

1. We replaced the ImageNet supervised model with the Swin MoCo model as the pre-trained model for Swin-Unet.
2. We used the AdamW optimizer and adjusted the learning rate accordingly.
3. We employed different data augmentations.

The final results outperform the performance of the official Swin-Unet code on the Synapse dataset. Our results not only show a slight improvement in Dice but also a significant improvement in HD. 

Running this project is expected to yield the following results:

```
[aorta] Dice: 84.15, HD: 10.79
[gallbladder] Dice: 65.33, HD: 10.91
[left_kidney] Dice: 86.91, HD: 8.97
[right_kidney] Dice: 81.76, HD: 13.58
[liver] Dice: 94.33, HD: 7.40
[pancreas] Dice: 57.86, HD: 12.13
[spleen] Dice: 88.16, HD: 14.37
[stomach] Dice: 79.01, HD: 16.99

[mean] Dice: 79.69, HD: 11.89
```

## Data preparation
The Synapse dataset used is identical to the Swin-Unet paper. 

Access to the synapse multi-organ dataset:
	1. Sign up in the [official Synapse website](https://www.synapse.org/#!Synapse:syn3193805/wiki/) and download the dataset. Convert them to numpy format, clip the images within [-125, 275], normalize each 3D image to [0, 1], and extract 2D slices from 3D volume for training cases while keeping the 3D volume in h5 format for testing cases.
	2. You can also send an Email directly to xuzian1113@foxmail.com to request the preprocessed data for reproduction.

The directory structure of the synapse dataset is as follows:

```bash
.
└──Synapse
   ├── test_vol_h5
   │   ├── case0001.npy.h5
   │   └── *.npy.h5
   └── train_npz
       ├── case0005_slice000.npz
       └── *.npz
```



## Usage
### upstream
1. Download all the code contained in the "upstream" folder and install the required environments in "requirements.txt".
2. Open "train.py" and fill in the "dataset-path". Example: 'Synapse/train_npz'.
3. Swin MoCo has two training approaches:
  - For training Swin MoCo from scratch, you need to set weights=None. 
  - For training Swin MoCo with transfer learning, you need to download [Swin Transformer's ImageNet-1K supervised model](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth) and set weights='swin_tiny_patch4_window7_224.pth'.
4. Run "train.py". If all goes well, you will get the pre-trained model "model.pth" after training.

### downstream

1. Download all the code contained in the "downstream" folder and install the required environments in "requirements.txt".
2. Open "train.py" and fill in the "dataset_path". Example: 'Synapse/train_npz'.
3. Fill in the "weights_path". Example: '../Swin MoCo/model/model.pth'.
4. Run "train.py". If all goes well, you will get the trained model "model-150.pth" after training.
5. Open "test.py" and fill in the "dataset_path". Example: 'Synapse/test_vol_h5'.
6. Fill in the "weights_path".  Example: 'trained_model/model-150.pth'.
7. Run "test.py". If all goes well, you will get the evaluation metrics recorded in "metric.txt".


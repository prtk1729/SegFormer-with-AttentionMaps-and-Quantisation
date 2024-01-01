# basic imports
import numpy as np

# DL library imports
import torch
import torch.nn as nn
import torch.nn.functional as F

# libraries for loading image, plotting
import cv2
import matplotlib.pyplot as plt

from einops import rearrange
import segmentation_models_pytorch as smp
from timm.models.layers import drop_path, trunc_normal_

from dataset import PrepareDataset


if __name__ == '__main__':
    prepare_dataset = PrepareDataset(8, 8, 0.8)
    train_dataloader, val_dataloader, test_dataloader = prepare_dataset.make_dataset_splits()
    
    # For train, test -> 8, 8
    # There are 2380 train images, 595 validation images, 500 test Images
    # Input shape = torch.Size([3, 512, 1024]), output label shape = torch.Size([512, 1024])



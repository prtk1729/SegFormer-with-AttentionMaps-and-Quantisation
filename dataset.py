# utility functions to get Cityscapes Pytorch dataset and dataloaders
from utils import get_dataloaders
from cityScapes_utils import get_cs_datasets

from utils import inverse_transform
from cityScapes_utils import train_id_to_color as cs_train_id_to_color
import numpy as np
import matplotlib.pyplot as plt

class PrepareDataset:
    def __init__(self, train_batch_size, test_batch_size, fraction=0.8):
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.fraction = fraction

    def make_dataset_splits(self):
            # splits
        # train folder -> 80% train, 20% test
        # test folder  -> val [completely new, not seen even by loss fn :p ]
        train_set, val_set, test_set = get_cs_datasets(rootDir="/home/prateek/ThinkAuto/data", fraction=self.fraction)
        sample_image, sample_label = train_set[0]
        print(f"There are {len(train_set)} train images, {len(val_set)} validation images, {len(test_set)} test Images")
        print(f"Input shape = {sample_image.shape}, output label shape = {sample_label.shape}")
        train_dataloader, val_dataloader, test_dataloader = get_dataloaders(train_set, val_set, test_set, train_batch_size=self.train_batch_size, test_batch_size=self.test_batch_size)
        
        return train_dataloader, val_dataloader, test_dataloader

        # There are 2380 train images, 595 validation images, 500 test Images
        # Input shape = torch.Size([3, 512, 1024]), output label shape = torch.Size([512, 1024])
        
        # show_sample_datapoint(train_set) # checked
        
    def show_sample_datapoint(train_set):
        rgb_image, label = train_set[np.random.choice(len(train_set))]
        rgb_image = inverse_transform(rgb_image).permute(1, 2, 0).cpu().detach().numpy()
        label = label.cpu().detach().numpy()

        # plot sample image
        fig, axes = plt.subplots(1,2, figsize=(20,10))
        axes[0].imshow(rgb_image);
        axes[0].set_title("Image");
        axes[0].axis('off');
        axes[1].imshow(cs_train_id_to_color[label]);
        axes[1].set_title("Label");
        axes[1].axis('off');
        return


    
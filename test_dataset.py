from dataset import PrepareDataset

if __name__ == '__main__':
    prepare_dataset = PrepareDataset(8, 8, 0.8)
    train_dataloader, val_dataloader, test_dataloader = prepare_dataset.make_dataset_splits()
    
    # For train, test -> 8, 8
    # There are 2380 train images, 595 validation images, 500 test Images
    # Input shape = torch.Size([3, 512, 1024]), output label shape = torch.Size([512, 1024])
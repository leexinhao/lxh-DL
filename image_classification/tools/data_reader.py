r"""
Module for loading data.
"""
import torch
from torch.utils.data import DataLoader
from torchvision import datasets



def load_FashionMnist(batch_size, valid_split=None, root_dir="Datasets", random_seed=233, download=False, transform=None, shuffle=True, drop_last=True, num_workers=0):
    r"""加载Fashion mnist图像分类数据集"""
    torch.manual_seed(random_seed)
    # Download or load training data from open datasets.
    train_data = datasets.FashionMNIST(
        root=root_dir,
        train=True,
        download=download,
        transform=transform,
    )
    valid_iter = None
    if valid_split is not None:
        assert valid_split > 0 and valid_split < 1, f"illegal valid+split: {valid_split}"
        valid_num = int(len(train_data) * valid_split)
        train_num = len(train_data) - valid_num
        train_data, valid_data = torch.utils.data.random_split(train_data, [train_num, valid_num])  
        valid_iter = DataLoader(valid_data, batch_size=batch_size, drop_last=drop_last, num_workers=num_workers)

    train_iter = DataLoader(train_data, batch_size=batch_size,
                            shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)
    


    # Download or load test data from open datasets.
    test_data = datasets.FashionMNIST(
        root=root_dir,
        train=False,
        download=download,
        transform=transform,
    )
    
    test_iter = DataLoader(test_data, batch_size=batch_size,
                           drop_last=drop_last, num_workers=num_workers)


    return train_iter, valid_iter, test_iter



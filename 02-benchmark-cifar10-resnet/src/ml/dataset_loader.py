"""Dataset loader.

- Author: Bono (bnabis93, github)
- Email: qhsh9713@gmail.com

Reference : https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
"""
from typing import Tuple

import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


def set_train_validate_loader(
    mean: Tuple[float, float, float] = (0.4914, 0.4822, 0.4465),
    std: Tuple[float, float, float] = (0.2023, 0.1994, 0.2010),
    validation_ratio=0.2,
    is_shuffled=True,
) -> Tuple[DataLoader, DataLoader]:
    """Set training and validation loader.
    Validation loader is splited by training dataset.
    """
    print("==> Preparing data..")
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    transform_valid = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean, std),]
    )

    train_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )

    validate_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_valid
    )

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(validation_ratio * num_train))

    if is_shuffled:
        np.random.seed(42)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = DataLoader(
        train_dataset, batch_size=128, sampler=train_sampler, num_workers=2
    )
    validate_loader = DataLoader(
        validate_dataset, batch_size=100, sampler=valid_sampler, num_workers=2
    )

    return train_loader, validate_loader


def set_test_loader(
    mean: Tuple[float, float, float] = (0.4914, 0.4822, 0.4465),
    std: Tuple[float, float, float] = (0.2023, 0.1994, 0.2010),
    is_shuffled=True,
) -> Tuple[DataLoader, DataLoader]:
    """Set test data loader."""
    print("==> Preparing data..")
    transform_test = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean, std),]
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )

    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=is_shuffled, num_workers=2
    )
    return test_loader

"""Functions for data handling.
- Author: Bono (bnabis93, github)
- Email: qhsh9713@gmail.com
"""


from typing import Tuple

import torch
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets, transforms


def get_dataloaders(
    use_cuda: bool, train_batch_size: int, valid_batch_size: int
) -> Tuple[DataLoader, DataLoader]:
    """Create data loader and return it."""
    train_kwargs = {"batch_size": train_batch_size}
    valid_kwargs = {"batch_size": valid_batch_size}
    if use_cuda:
        cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
        train_kwargs.update(cuda_kwargs)
        valid_kwargs.update(cuda_kwargs)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [50000, 10000])

    # test_dataset = datasets.MNIST("../data", train=False, transform=transform)
    # test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)
    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    val_dataset = torch.utils.data.DataLoader(train_dataset, **valid_kwargs)

    return train_loader, val_dataset

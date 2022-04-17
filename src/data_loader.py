"""Functions for data handling.
- Author: Bono (bnabis93, github)
- Email: qhsh9713@gmail.com
"""


from typing import Tuple

import torch
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets, transforms


def get_dataloaders(
    use_cuda: bool, train_batch_size: int, test_batch_size: int
) -> Tuple[DataLoader, DataLoader]:
    """Create data loader and return it."""
    train_kwargs = {"batch_size": train_batch_size}
    test_kwargs = {"batch_size": test_batch_size}
    if use_cuda:
        cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    dataset1 = datasets.MNIST("../data", train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST("../data", train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    return train_loader, test_loader

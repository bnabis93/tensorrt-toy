"""Basic mnist dataset training code.

from https://github.com/pytorch/etensoramples/blob/main/mnist/main.py
- Author: bono
- Email: qhsh9713@gmail.com
"""
from __future__ import print_function

import argparse
from typing import Any, Dict

import torch
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.dataloader import DataLoader

from ml.data import get_dataloaders
from ml.model import ConvNet

parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
parser.add_argument(
    "--batch-size",
    type=int,
    default=64,
    metavar="N",
    help="input batch size for training (default: 64)",
)
parser.add_argument(
    "--epochs",
    type=int,
    default=14,
    metavar="N",
    help="number of epochs to train (default: 14)",
)
parser.add_argument(
    "--valid-batch-size",
    type=int,
    default=1000,
    metavar="N",
    help="input batch size for validation (default: 1000)",
)
parser.add_argument(
    "--lr", type=float, default=1.0, metavar="LR", help="learning rate (default: 1.0)"
)
parser.add_argument(
    "--gamma",
    type=float,
    default=0.7,
    metavar="M",
    help="Learning rate step gamma (default: 0.7)",
)
parser.add_argument(
    "--no-cuda", action="store_true", default=False, help="disables CUDA training"
)
parser.add_argument(
    "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
)
parser.add_argument(
    "--log-interval",
    type=int,
    default=10,
    metavar="N",
    help="how many batches to wait before logging training status",
)
parser.add_argument(
    "--save-model",
    action="store_true",
    default=False,
    help="For Saving the current Model",
)
args = parser.parse_args()


def train(
    model: torch.nn.Module,
    device: torch.device,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    epoch: int,
    param: Dict[Any, Any],
) -> None:
    """Model training."""
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % param.log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
    return


def validation(
    model: torch.nn.Module, valid_loader: DataLoader, device: torch.device
) -> None:
    """Get the valid accuracy."""
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in valid_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += F.nll_loss(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    val_loss /= len(valid_loader.dataset)

    print(
        "\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            val_loss,
            correct,
            len(valid_loader.dataset),
            100.0 * correct / len(valid_loader.dataset),
        )
    )


def main():
    """main function for training using mnist dataset."""
    torch.manual_seed(args.seed)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = get_dataloaders(
        use_cuda=use_cuda,
        train_batch_size=args.batch_size,
        valid_batch_size=args.valid_batch_size,
    )

    model = ConvNet().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(
            model=model,
            device=device,
            train_loader=train_loader,
            optimizer=optimizer,
            epoch=epoch,
            param=args,
        )
        validation(model=model, valid_loader=val_loader, device=device)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == "__main__":
    main()

"""Training cifar10 dataset.

- Author: Bono (bnabis93, github)
- Email: qhsh9713@gmail.com

Reference : https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
"""
import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader

from ml.dataset_loader import set_train_validate_loader
from ml.model import resnet34
from utils import progress_bar

parser = argparse.ArgumentParser(description="PyTorch CIFAR10 Training")
parser.add_argument("--lr", default=0.1, type=float, help="learning rate")
parser.add_argument("--epoch", default=150, type=int, help="validation dataset ratio")
parser.add_argument(
    "--valid_size", default=0.2, type=float, help="validation dataset ratio"
)
parser.add_argument(
    "--shuffle", default=True, type=bool, help="is dataset shuffled?",
)
args = parser.parse_args()

best_acc = 0.0  # best test accuracy

# Training
def train(
    net: torch.nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    epoch: int,
    device: torch.device,
) -> None:
    "Train the model for classify the cifar10 dataset."
    print("\nEpoch: %d" % epoch)
    net.train()
    criterion = nn.CrossEntropyLoss()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(
            batch_idx,
            len(train_loader),
            "Loss: %.3f | Acc: %.3f%% (%d/%d)"
            % (train_loss / (batch_idx + 1), 100.0 * correct / total, correct, total),
        )


def validate(
    net: torch.nn.Module, validate_loader: DataLoader, epoch: int, device: torch.device,
) -> None:
    "Valid the model for classify the cifar10 dataset."
    global best_acc
    net.eval()
    valid_loss = 0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(validate_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            valid_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(
                batch_idx,
                len(validate_loader),
                "Loss: %.3f | Acc: %.3f%% (%d/%d)"
                % (
                    valid_loss / (batch_idx + 1),
                    100.0 * correct / total,
                    correct,
                    total,
                ),
            )

    # Save checkpoint.
    acc = 100.0 * correct / total
    if acc > best_acc:
        print("Saving..")
        if not os.path.isdir("output"):
            os.mkdir("output")
        torch.save(net.state_dict(), "./output/ckpt.pth")
        best_acc = acc


def main():
    """Main function for training and validation."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    epochs = args.epoch
    print("device : ", device)
    # Set data loader
    train_loader, validate_loader = set_train_validate_loader(
        validation_ratio=args.valid_size, is_shuffled=args.shuffle
    )

    # Model
    print("==> Building model..")
    net = resnet34()
    net = net.to(device)
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        train(
            net=net,
            train_loader=train_loader,
            optimizer=optimizer,
            epoch=epoch,
            device=device,
        )
        validate(
            net=net, validate_loader=validate_loader, epoch=epoch, device=device,
        )
        scheduler.step()


if __name__ == "__main__":
    main()

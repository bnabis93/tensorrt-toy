import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader

from ml.dataset_loader import set_test_loader
from ml.model import resnet34
from utils import progress_bar

parser = argparse.ArgumentParser(description="PyTorch CIFAR10 Training")
parser.add_argument(
    "--model_path",
    default="./checkpoint/ckpt.pth",
    type=str,
    help="Pretrained model weight path",
)
args = parser.parse_args()


def test(net: torch.nn.Module, test_loader: DataLoader, device: torch.device,) -> float:
    "Valid the model for classify the cifar10 dataset."
    net.eval()
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            _ = criterion(outputs, targets)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(
                batch_idx,
                len(test_loader),
                f"Acc: {100.0 * correct / total} ({correct}/{total})",
            )

    # Save checkpoint.
    acc = 100.0 * correct / total
    return acc


def main():
    """Main function for benchmarking."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_loader = set_test_loader()
    net = resnet34()
    net = net.to(device)
    if device == "cuda":
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    checkpoint = torch.load(args.model_path)
    print("checkpoint : ", checkpoint.keys())
    net.load_state_dict(checkpoint["net"])
    test_acc = test(net=net, device=device, test_loader=test_loader)
    print(f"Cifar 10 pytorch model test acc : {test_acc}")


if __name__ == "__main__":
    main()

"""Benchmark torch model vs trt model in cifar10 test dataset.

- Author: Bono (bnabis93, github)
- Email: qhsh9713@gmail.com
"""
import argparse
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader

from ml.dataset_loader import set_test_loader
from ml.model import resnet34
from trt.trt_infer import TrtModel
from utils import progress_bar

parser = argparse.ArgumentParser(description="PyTorch CIFAR10 Training")
parser.add_argument(
    "--model_path",
    default="./output/ckpt.pth",
    type=str,
    help="Pretrained model weight path",
)
parser.add_argument(
    "--trt_path",
    default="./output/model.trt",
    type=str,
    help="Pretrained model weight path",
)
args = parser.parse_args()


def test(net: torch.nn.Module, test_loader: DataLoader, device: torch.device,) -> float:
    "Test the model for classify the cifar10 dataset."
    net.eval()
    correct = 0
    total = 0
    infernce_speeds = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            start = time.time()
            outputs = net(inputs)
            infernce_speeds.append(time.time() - start)
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
    avg_inference_speed = sum(infernce_speeds) / len(infernce_speeds)
    return acc, avg_inference_speed


def trt_test(
    trt_net, test_loader,
):
    "Valid the model for classify the cifar10 dataset."
    correct = 0
    total = 0
    infernce_speeds = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.numpy(), targets.numpy()
            start = time.time()
            outputs = trt_net(inputs)
            infernce_speeds.append(time.time() - start)
            predicted = np.argmax(outputs)
            total += 1
            if predicted == targets[0]:
                correct += 1
            progress_bar(
                batch_idx,
                len(test_loader),
                f"Acc: {100.0 * correct / total} ({correct}/{total})",
            )

    # Save checkpoint.
    acc = 100.0 * correct / total
    avg_inference_speed = sum(infernce_speeds) / len(infernce_speeds)
    return acc, avg_inference_speed


def main():
    """Main function for benchmarking."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_loader = set_test_loader()
    net = resnet34()
    net = net.to(device)
    net.load_state_dict(torch.load(args.model_path))
    test_acc, test_inference_speed = test(
        net=net, device=device, test_loader=test_loader
    )

    trt_engine_path = args.trt_path
    trt_net = TrtModel(trt_engine_path)
    trt_test_acc, trt_test_inference_speed = trt_test(trt_net, test_loader)
    print(f"Torch acc : {test_acc} TensorRT acc : {trt_test_acc}")
    print(
        f"Torch inference speed per image : {round(test_inference_speed * 1000, 5)}ms \
        TensorRT inference speed per image : {round(trt_test_inference_speed * 1000, 5)}ms"
    )


if __name__ == "__main__":
    main()

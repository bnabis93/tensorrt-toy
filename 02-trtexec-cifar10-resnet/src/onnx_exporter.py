import argparse
import os

import torch

from ml.model import resnet34

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="./output/ckpt.pth")
    parser.add_argument("--save", default="model.onnx")
    args = parser.parse_args()

    resnet34 = resnet34()
    resnet34.load_state_dict(torch.load(args.model_path))
    dummy_input = torch.randn(1, 3, 32, 32)
    resnet34 = resnet34.eval()

    torch.onnx.export(
        resnet34, dummy_input, args.save, opset_version=12,
    )

    print("Saved {}".format(args.save))

import argparse

import torch
import torchvision.models as models

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", default="model.onnx")
    args = parser.parse_args()

    resnet50 = models.resnet50(pretrained=True)
    dummy_input = torch.randn(1, 3, 224, 224)
    resnet50 = resnet50.eval()

    torch.onnx.export(
        resnet50,
        dummy_input,
        args.save,
        opset_version=12,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"},},
    )

    print("Saved {}".format(args.save))

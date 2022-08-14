import argparse
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch_tensorrt
import torchvision.models as models
from torch_tensorrt import compile

parser = argparse.ArgumentParser(description="PyTorch CIFAR10 Training")
parser.add_argument(
    "--batch_size", default=1, type=int, help="Benchmark batchsize",
)
parser.add_argument(
    "--num_run", default=1000, type=int, help="Number of run in benchmark.",
)
parser.add_argument(
    "--quant", default="fp32", type=str, help="Set the quantization.",
)
args = parser.parse_args()
cudnn.benchmark = True


def benchmark(
    model, input_shape=(1024, 1, 224, 224), dtype="fp32", nwarmup=50, nruns=10000
):
    """Benchmark module."""
    input_data = torch.randn(input_shape)
    input_data = input_data.to("cuda")
    if dtype == "fp16":
        input_data = input_data.half()

    print("Warm up ...")
    with torch.no_grad():
        for _ in range(nwarmup):
            features = model(input_data)
    torch.cuda.synchronize()
    print("Start timing ...")
    timings = []
    with torch.no_grad():
        for i in range(1, nruns + 1):
            start_time = time.time()
            features = model(input_data)
            torch.cuda.synchronize()
            end_time = time.time()
            timings.append(end_time - start_time)
            if i % 10 == 0:
                print(
                    "Iteration %d/%d, ave batch time %.2f ms"
                    % (i, nruns, np.mean(timings) * 1000)
                )

    print("Input shape:", input_data.size())
    print("Output features size:", features.size())
    print("Average batch time: %.2f ms" % (np.mean(timings) * 1000))


def main():
    """Main."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    resnet50 = models.resnet50(pretrained=True)
    resnet50.eval().to(device)
    print("Torch benchmark")
    # benchmark(
    #     model=resnet50, input_shape=(args.batch_size, 3, 224, 224), nruns=args.num_run,
    # )
    
    if args.quant == "fp32":
        print("TensorRT FP32 benchmark")
        trt_model_fp32 = compile(
            resnet50,
            inputs=[
                torch_tensorrt.Input(
                    (args.batch_size, 3, 224, 224), dtype=torch.float32
                )
            ],
            enabled_precisions=torch.float32,  # Run with FP32
            workspace_size=1 << 22,
        )
        benchmark(
            model=trt_model_fp32,
            input_shape=(args.batch_size, 3, 224, 224),
            nruns=args.num_run,
        )
    elif args.quant == "fp16":
        print("TensorRT FP16 benchmark")
        trt_model_fp16 = compile(
            resnet50,
            inputs=[
                torch_tensorrt.Input((args.batch_size, 3, 224, 224), dtype=torch.half)
            ],
            enabled_precisions={torch.half},  # Run with FP32
            workspace_size=1 << 22,
        )
        benchmark(
            model=trt_model_fp16,
            input_shape=(args.batch_size, 3, 224, 224),
            nruns=args.num_run,
        )
    else:
        ValueError("Unsupported quantization type.")


if __name__ == "__main__":
    main()

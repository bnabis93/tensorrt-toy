# Torch-tensorrt basic usage
- I used [NGC Pytorch image](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch) to tensorrt converting and benchmarking.

## Core concepts
> **Torch-TensorRT is a compiler for PyTorch/TorchScript**, targeting NVIDIA GPUs via NVIDIA’s TensorRT Deep Learning Optimizer and Runtime. Unlike PyTorch’s Just-In-Time (JIT) compiler, **Torch-TensorRT is an Ahead-of-Time (AOT) compiler**, meaning that before you deploy your TorchScript code, you go through an explicit compile step to convert a standard TorchScript program into an module targeting a TensorRT engine. Torch-TensorRT operates **as a PyTorch extention and compiles modules** that **integrate into the JIT runtime seamlessly.** After compilation using the optimized graph should feel no different than running a TorchScript module. You also have access to TensorRT’s suite of configurations at compile time, so you are able to specify operating precision (FP32/FP16/INT8) and other settings for your module. - [ref](https://nvidia.github.io/Torch-TensorRT/)



## Environment
- Ubuntu 20.04
- GPU : RTX 3080



## Reference
- Torch-TensorRT : https://github.com/NVIDIA/Torch-TensorRT
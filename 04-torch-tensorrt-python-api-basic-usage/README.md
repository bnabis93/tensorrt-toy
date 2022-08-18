# 04-torch-tensorrt-python-api-basic-usage
- Torch-TensorRT is now an official part of the PyTorch ecosystem and now available on PyTorch GitHub and Documentation. - [link](https://github.com/pytorch/TensorRT)

# Description
## Core Concepts
> Ahead of Time (AOT) compiling for PyTorch JIT and FX
- Torch-TensorRT is a **compiler** for PyTorch/TorchScript/FX, targeting NVIDIA GPUs via NVIDIA's TensorRT Deep Learning Optimizer and Runtime. 
    - Ahead-of-Time (AOT) compiler. (Not PyTorch's Just-In-Time (JIT) compiler.)
- Ahead-of-Time (AOT) compiler vs Just-In-Time (JIT) compiler
    - Ahead-of-Time (AOT) compiler : Explicit compile before excute program.
    - Just-In-Time (JIT) compiler : Compile when program was excuted time. (Dynamic compile)
- Torch-TensorRT operates as a PyTorch extention and compiles modules that **integrate into the JIT runtime seamlessly.**

## Environment
- Ubuntu 20.04
- NGC-Pytorch container (ver 22.07)

## Reference
- [Pytorch official documentation about torch-tensorrt](https://pytorch.org/TensorRT/tutorials/installation.html)

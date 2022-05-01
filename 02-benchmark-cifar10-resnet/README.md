# Torch model vs TensorRT model benchmark using cifar10 dataset and resnet model.
- I used [NGC Pytorch image](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch) to tensorrt converting and benchmarking.

## Environment
- Ubuntu 20.04
- GPU : RTX 3080

## Prerequisites
- Install [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)
```
$ make env
$ conda activate 02-benchmark-cifar10-resnet
$ make setup
```

## How to play
### 1. Training cifar10 dataset using resnet 
```
$ make train
$ ls output
ckpt.pth
```

### 2. Convert torch resnet model to tensorrt model
- I used [trtexec](https://github.com/NVIDIA/TensorRT/tree/main/samples/trtexec) to convert torch model to tensorrt model. 
```
$ make convert-trt
$ ls output
ckpt.pth    model.trt
```

### 3. Benchmark cifar10 dataset 
```
$ make benchmark

...
 [============================ 10000/10000 =======================>]  Step: 4ms | Tot: 44s170ms | Acc: 94.9 (9490/10000)
[05/01/2022-16:00:00] [TRT] [W] Using an engine plan file across different models of devices is not recommended and is likely to affect performance or even cause errors.
 [============================ 10000/10000 =======================>]  Step: 2ms | Tot: 21s279ms | Acc: 94.9 (9490/10000)

Torch acc : 94.9 TensorRT acc : 94.9
Torch inference speed per image : 3.8048ms         TensorRT inference speed per image : 1.52217ms

```

## Reference
- Cifar10 training code : https://github.com/kuangliu/pytorch-cifar/
- TensorRT infernce code : https://stackoverflow.com/questions/59280745/inference-with-tensorrt-engine-file-on-python
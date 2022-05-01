# trtexec mnist dataset with resnet example

## Environment
- Ubuntu 20.04
- GPU : RTX 3080 
- Conda

## Prerequisites
```
$ make env
$ conda activate 02-trtexec-cifar10-resnet
$ make setup
```

## Training cifar10 dataset using resnet 
```
$ make train
$ ls output
ckpt.pth
```

## Convert torch resnet model to tensorrt model
```
$ make convert-trt
$ ls output
ckpt.pth    model.trt
```

## Benchmark cifar10 dataset 
```
$ make benchmark

Files already downloaded and verified
 [============================ 10000/10000 =======================>]  Step: 4ms | Tot: 44s170ms | Acc: 94.9 (9490/10000)
[05/01/2022-16:00:00] [TRT] [W] Using an engine plan file across different models of devices is not recommended and is likely to affect performance or even cause errors.
 [============================ 10000/10000 =======================>]  Step: 2ms | Tot: 21s279ms | Acc: 94.9 (9490/10000)

Torch acc : 94.9 TensorRT acc : 94.9
Torch inference speed per image : 3.8048ms         TensorRT inference speed per image : 1.52217ms

```

## Reference
- Cifar10 train example : https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
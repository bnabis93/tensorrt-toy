# trtexec mnist dataset with resnet example

## Environment
- Ubuntu 20.04
- GPU : RTX 3080 
- Conda

## Prerequisites
```
make env
conda activate 02-trtexec-cifar10-resnet
make setup
```

## Training cifar10 dataset using resnet 
```
make train
```

## Convert torch resnet model to tensorrt model
```
make convert-trt
```

## Benchmark cifar10 dataset 


## Reference
- Cifar10 train example : https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
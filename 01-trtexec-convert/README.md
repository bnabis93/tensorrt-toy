# trtexec-convert example

## 1. Create trt model using ImageNet-pretrained resnet50
- ImageNet-pretrained resnet50 from [torchvision](https://pytorch.org/vision/stable/models.html)
```bash
make model
```

## 2. trt model benchmarking
- batchsize = 1
```bash
make benchmark
```

## Reference
- https://github.com/NVIDIA/TensorRT/tree/main/samples/trtexec#building-trtexec

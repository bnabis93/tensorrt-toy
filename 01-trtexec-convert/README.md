# trtexec-convert example

## 1. Create trt model using ImageNet-pretrained resnet50
- ImageNet-pretrained resnet50 from [torchvision](https://pytorch.org/vision/stable/models.html)
```bash
$ make model
```

## 2. trt model benchmarking
- batchsize = 1
```bash
$ make benchmark
# Then, you can get performance summary
=== Performance summary ===
Throughput: 667.181 qps
Latency: min = 1.48242 ms, max = 1.59076 ms, mean = 1.49393 ms, median = 1.48682 ms, percentile(99%) = 1.57001 ms
End-to-End Host Latency: min = 1.49353 ms, max = 3.01935 ms, mean = 2.84119 ms, median = 2.85034 ms, percentile(99%) = 3.01239 ms

...

&&&& PASSED TensorRT.trtexec [TensorRT v8203] # trtexec --loadEngine=.//output/model.trt --batch=1
```

## Reference
- https://github.com/NVIDIA/TensorRT/tree/main/samples/trtexec#building-trtexec

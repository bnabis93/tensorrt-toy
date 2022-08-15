# Speeding Up Deep Learning Inference Using NVIDIA TensorRT 
## Objective
- Increasing throughput and reducing latency during inference.
    - throughput : Number of inference per second.
    - latency : The time it takes to execute a just one inference.

## Simple TensorRT example
- Flow : torch model -> onnx -> tensorrt -> apply optimizations and generate engine -> evalute tensorrt model inference speed on GPU.
    - ONNX is a standard for representing deep learning models enabling them to be transferred between frameworks.
    - ONNX has a `protobuf` dependency. (.pb means that protobuf) - [link](https://github.com/onnx/onnx-tensorrt)
- TensorRT components 
    - ONNX parser : parsing ONNX models into a TensorRT network definition. - [link](https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/parsers/Onnx/pyOnnx.html)
    - Builder : Input : tensorrt, output : Target gpu optimized engine.
    - Engine : Engine perform the inference. Input : data, output : inference output. 
    - Logger : Logger in build and inference phases. (builder / engine)

## Import the ONNX model into TensorRT, generate the engine, and perform inference
```
 // About tensorrt (builder phase)
 // Declare the CUDA engine
 SampleUniquePtr<nvinfer1::ICudaEngine> mEngine{nullptr};
 ...
 // Create the CUDA engine
 mEngine = SampleUniquePtr<nvinfer1::ICudaEngine>   (builder->buildEngineWithConfig(*network, *config));
```
- `SimpleOnnx::createEngine` : ONNX model as input for create engine. 
- `SimpleOnnx::buildEngine` : Parses the ONNX model, save onnx information to network object.
- If you want to use dynamic input -> use builder class. (Should optimize builder class)
- What is the optimization in tensorrt pipielint?
    - Optimum input, minimum, and maximum dimensions. 
    - Build select the kernel in runtime.
    - That is, builder class set the inference hyperparameter like batch size, input size, min / max dims. (This is called 'optimization')
    - So, tensorrt already set this inference hyperparameter. (Bcz builder create tensorrt not engine.)
- Engine phase, create context for inference.
```
// About engine (engine)
// Declare the execution context
 SampleUniquePtr<nvinfer1::IExecutionContext> mContext{nullptr};
 ...
 // Create the execution context
 mContext = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
```
- For inference, inputs are copied from **host (CPU) to device (GPU)** (Has a queue called enqueueV2)
    - enqueue does some CPU work to prepare for GPU kernel launches. - [enqueue issue](https://github.com/NVIDIA/TensorRT/issues/999)
    - enqueueV2 request to CUDA Stream / Determine input runtime batch size / Determine pointers to input and output / Determine CUDA stream to be used for kernel execution. 
    - We can set inference requests on the GPU asynchronously in context.

## Batch your inputs
- Real applications commonly **batch inputs**(Not single input) to achieve higher performance and efficiency. 
    - Batch input can be computed in parallel.
    - Larger batches generally enable **more efficient use of GPU resources.**

## Profile the application (Measure its performance.)
- latency, throughput, ...
- Consider the following information when evaluate latency.
    - Transfer data between the GPU and CPU before inference initiates and after inference completes.
-  Pre-fetch data to the GPU + overlap compute with data + hide data transfer overhead. -> `cudaEventElapsedTime`
- `cudaEventElapsedTime` : Computes the elapsed time between two events. [link](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html#group__CUDART__EVENT_1g40159125411db92c835edb46a0989cd6)
    - CudaEventRecord() operation takes place asynchronously and there is no guarantee that the measured latency is actually just between the two events.

## Optimize your application
- Best Practices for TensorRT Performance. - [link](https://docs.nvidia.com/deeplearning/tensorrt/)
    - Use mixed precision computation.
    - Change the workspace size.
    - Reuse the TensorRT engine. (Keep it in GPU memory?)

### Use mixed precision computation
- Can use FP16 and INT8 precision for inference (default : FP32)
- Also **mix computations** in FP32 and FP16 precision. (FP32 + FP16 / FP16 + INT8 / FP32 + INT8, ...)

### Change the workspace size
- Increase resource. 
- Could share the GPU at the same time. 

### Reuse the TensorRT engine
- Serializing the engine (Reduce the pipeline process.)

## Reference
- https://developer.nvidia.com/blog/speeding-up-deep-learning-inference-using-tensorrt-updated/
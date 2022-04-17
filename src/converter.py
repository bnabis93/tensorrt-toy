import torch
import torchvision.models as models
from torch2trt import torch2trt

resnet18 = models.resnet18(pretrained=True).cuda()
torch.save(resnet18.state_dict(), "./resnet18.pt")

dummy_input = torch.randn(1, 3, 224, 224, device="cuda")
# convert to ONNX format
torch.onnx.export(
    resnet18,
    dummy_input,
    "./resnet18.onnx",
    input_names=["input"],
    output_names=["output"],
)

# convert to TensorRT feeding sample data as input
resnet18_trt = torch2trt(resnet18, [dummy_input])
# can save the model as tensorrt engine.
with open("./resnet18.engine", "wb") as f:
    f.write(resnet18_trt.engine.serialize())

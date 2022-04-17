import time

import torch
import torchvision.models as models
from torch2trt import torch2trt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# create some regular pytorch model.
resnet = models.resnet18()
resnet.load_state_dict(torch.load("./resnet18.pt", map_location=device))


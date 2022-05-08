import json
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.models as models
from PIL import Image
from torchvision import transforms

cudnn.benchmark = True


def rn50_preprocess():
    """Preprocess module."""
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return preprocess


def predict(img_path, model, description):
    """Predict module.
    decode the results into ([predicted class, description], probability)."""
    img = Image.open(img_path)
    preprocess = rn50_preprocess()
    input_tensor = preprocess(img)
    input_batch = input_tensor.unsqueeze(
        0
    )  # create a mini-batch as expected by the model

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to("cuda")
        model.to("cuda")

    with torch.no_grad():
        output = model(input_batch)
        # Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
        sm_output = torch.nn.functional.softmax(output[0], dim=0)

    ind = torch.argmax(sm_output)
    return (
        description[str(ind.item())],
        sm_output[ind],
    )  # ([predicted class, description], probability)


def benchmark(
    model, input_shape=(1024, 1, 224, 224), dtype="fp32", nwarmup=50, nruns=10000
):
    """Benchmark module."""
    input_data = torch.randn(input_shape)
    input_data = input_data.to("cuda")
    if dtype == "fp16":
        input_data = input_data.half()

    print("Warm up ...")
    with torch.no_grad():
        for _ in range(nwarmup):
            features = model(input_data)
    torch.cuda.synchronize()
    print("Start timing ...")
    timings = []
    with torch.no_grad():
        for i in range(1, nruns + 1):
            start_time = time.time()
            features = model(input_data)
            torch.cuda.synchronize()
            end_time = time.time()
            timings.append(end_time - start_time)
            if i % 10 == 0:
                print(
                    "Iteration %d/%d, ave batch time %.2f ms"
                    % (i, nruns, np.mean(timings) * 1000)
                )

    print("Input shape:", input_data.size())
    print("Output features size:", features.size())
    print("Average batch time: %.2f ms" % (np.mean(timings) * 1000))


def main():
    """Main."""
    for i in range(3):
        img_path = "./data/img%d.JPG" % i
        resnet50 = models.resnet50(pretrained=True)
        resnet50.eval()
        with open("./data/imagenet_class_index.json") as json_file:
            description = json.load(json_file)
        pred, prob = predict(img_path, resnet50, description)
        print("{} - Predicted: {}, Probablility: {}".format(img_path, pred, prob))


if __name__ == "__main__":
    main()

import time

import torch
import numpy as np
from torchvision import models, transforms
from torchvision.models.quantization import MobileNet_V2_QuantizedWeights

import cv2
from PIL import Image

torch.backends.quantized.engine = 'qnnpack'

cam = cv2.VideoCapture(0)

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

with open("imagenet_classes.txt", "r") as f:
    classes = [s.strip() for s in f.readlines()]

net = models.quantization.mobilenet_v2(weights=MobileNet_V2_QuantizedWeights.IMAGENET1K_QNNPACK_V1, quantize=True)

with torch.no_grad():
    while True:
        # read frame
        ret, image = cam.read()
        if not ret:
            raise RuntimeError("failed to read frame")

        # convert opencv output from BGR to RGB
        image = image[:, :, [2, 1, 0]]
        permuted = image

        # preprocess
        input_tensor = preprocess(image)

        # create a mini-batch as expected by the model
        input_batch = input_tensor.unsqueeze(0)

        # run model
        output = net(input_batch)
        # do something with output ...
        top = list(enumerate(output[0].softmax(dim=0)))
        top.sort(key=lambda x: x[1], reverse=True)
        for idx, val in top[:10]:
            print(f"{val.item()*100:.2f}% {classes[idx]}")

        # Display the resulting frame
        cv2.imshow('Video Test', image)

        key = cv2.waitKey(1) & 0xFF

        # Press 'q' key to break the loop
        if key == ord("q"):
            break

    # When everything done, release the capture
    cam.release()
    cv2.destroyAllWindows()
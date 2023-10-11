import torch
from torchvision import transforms
import cv2

# ---------------------------------------
# Setup and Util
# ---------------------------------------

# Initializing webcam
cam = cv2.VideoCapture(0)

# Reading in ImageNet classes
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]

model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
model.eval()

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ---------------------------------------
# Main Loop
# ---------------------------------------

with torch.no_grad():
    while True:
        # Read frame in from webcam
        ret, image = cam.read()

        # Process the image for mobilenet
        input_tensor = preprocess(image)
        input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

        # move the input and model to GPU for speed if available
        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')
            model.to('cuda')

        output = model(input_batch)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)


        # Show top categories per image
        top5_prob, top5_catid = torch.topk(probabilities, 5)
        for i in range(top5_prob.size(0)):
            print(categories[top5_catid[i]], top5_prob[i].item())

        # Display the resulting frame
        cv2.imshow('Video Test', image)

        key = cv2.waitKey(1) & 0xFF

        # Press 'q' key to break the loop
        if key == ord("q"):
            break

    # When everything done, release the capture
    cam.release()
    cv2.destroyAllWindows()
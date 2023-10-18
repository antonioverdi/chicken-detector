# IMPORTS
import RPi.GPIO as GPIO          
from time import sleep
import torch
from torchvision import transforms
import cv2

# ---------------------------------------
# Initializing GPIO 
# ---------------------------------------

in1 = 24
in2 = 23
en = 25
temp1 = 1
hall1 = 6
hall2 = 5


GPIO.setmode(GPIO.BCM)
GPIO.setup(in1,GPIO.OUT)
GPIO.setup(in2,GPIO.OUT)
GPIO.setup(en,GPIO.OUT)
GPIO.setup(hall1,GPIO.IN) 
GPIO.setup(hall2,GPIO.IN) 
GPIO.output(in1,GPIO.LOW)
GPIO.output(in2,GPIO.LOW)
p=GPIO.PWM(en,1000)
p.start(25)

# ---------------------------------------
# Initializing model and webcam 
# ---------------------------------------
cam = cv2.VideoCapture(0)

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

targets = {'hen', 'cock'}

# UTILITY FUNCTIONS

def door_up():
    GPIO.output(in1,GPIO.HIGH)
    GPIO.output(in2,GPIO.LOW)

def door_down():
    GPIO.output(in1,GPIO.LOW)
    GPIO.output(in2,GPIO.HIGH)

def door_stop():
    GPIO.output(in1,GPIO.LOW)
    GPIO.output(in2,GPIO.LOW)

def chicken_detected(cam, preprocess, model, targets):
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
    detected = set()
    for i in range(top5_prob.size(0)):
        detected.add(categories[top5_catid[i]])
    
    if len(detected.intersection(targets)) > 0:
        return True

def safe_kill():
        print('Performing safe shutoff!')
        GPIO.output(in1,False)
        GPIO.output(in2,False)
        GPIO.cleanup()



# DOOR OPERATING CODE
def run_door():
    BottomHall=GPIO.input(5)
    TopHall=GPIO.input(6)
    if BottomHall==0:print('Door is locked')
    if TopHall==0:print('Door is open')
    if BottomHall==0: #Door is locked
            print('The door is locked!')
            print('The door is going up!')
            while TopHall==1:
                    door_up()
                    TopHall=GPIO.input(6)
            if TopHall==0:
                    print('Door is open!')
                    door_stop()
    elif TopHall==0: #Door is open
            print('The door is open!')
            print('The door is going down!')
            while BottomHall==1:
                    door_down()
                    BottomHall=GPIO.input(5)
            if BottomHall==0:
                    print('Door is locked!')
                    door_stop()


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
        detected = set()
        for i in range(top5_prob.size(0)):
            detected.add(categories[top5_catid[i]])
        
        if len(detected.intersection(targets)) > 0:
            print("Chicken Detected")
            run_door()

        # Display the resulting frame
        cv2.imshow('Video Test', image)

        key = cv2.waitKey(1) & 0xFF

        # Press 'q' key to break the loop
        if key == ord("q"):
            break

    # When everything done, release the capture
    cam.release()
    cv2.destroyAllWindows()

safe_kill()

# When everything done, release the capture
cam.release()
cv2.destroyAllWindows()
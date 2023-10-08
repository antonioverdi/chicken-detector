from imutils.video import VideoStream
from imutils.video import FPS

import tensorflow as tf
import keras
from keras import datasets, layers, models
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import numpy as np
import argparse
import imutils
import time
import cv2

import matplotlib.pyplot as plt

# Trains a CNN on CIFAR10
def train_network(save_name, num_epochs):
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

    # Normalize pixel values to be between 0 and 1
    train_images, test_images = train_images / 255.0, test_images / 255.0

    # Creating the a sequential network architecture
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10))

    model.compile(optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])

    history = model.fit(train_images, train_labels, epochs=num_epochs, 
                    validation_data=(test_images, test_labels))
    
    model.save(save_name)



"""
-------------------------------------------------------------------------------------------------------------------------
Running video stream and inference
-------------------------------------------------------------------------------------------------------------------------

"""

loaded_model = keras.models.load_model("model20.keras")

print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
# warm up the camera for a couple of seconds
time.sleep(2.0)
fps = FPS().start()

while True:
    frame = vs.read()
    resized_image = (cv2.resize(frame, (32, 32))/255).reshape(-1, 32, 32, 3)
    preds = loaded_model.predict(resized_image)
    preds = np.argmax(preds, axis = 1)
    print(preds)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1) & 0xFF

	# Press 'q' key to break the loop
    if key == ord("q"):
        break

	# update the FPS counter
    fps.update()

# stop the timer
fps.stop()

# Display FPS Information: Total Elapsed time and an approximate FPS over the entire video stream
print("[INFO] Elapsed Time: {:.2f}".format(fps.elapsed()))
print("[INFO] Approximate FPS: {:.2f}".format(fps.fps()))

# Destroy windows and cleanup
cv2.destroyAllWindows()
# Stop the video stream
vs.stop()
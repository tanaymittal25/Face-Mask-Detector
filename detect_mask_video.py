#imports
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream

import numpy as np
import argparse
import imutils
import time
import cv2
import os

#Predict Function
def detect_and_predict(frame, face_net, mask_net):

    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

    face_net.setInput(blob)
    detections = face_net.forward()

    faces = []
    locs = []
    prediction = []

    for i in range(0, detections.shape[2]):

        confidence = detections[0, 0, i, 2]

        if confidence > args["confidence"]:

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (start_x, start_y, end_x, end_y) = box.astype("int")

            (start_x, start_y) = (max(0, start_x), max(0, start_y))
            (end_x, end_y) = (min(w - 1, end_x), min(h - 1, end_y))

            face = frame[start_y: end_y, start_x: end_x]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis = 0)

            faces.append(face)
            locs.append((start_x, start_y, end_x, end_y))

    if len(faces) > 0:
        prediction = mask_net.predict(faces)

    return (locs, prediction)

#ArgumentParser
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type = str, default = "face_detector", help = "Path to Model Dir")
ap.add_argument("-m", "--model", type = str, default = "mask_detector.model", help = "Path to Trained model")
ap.add_argument("-c", "--confidence", type = float, default = 0.5, help = "Minimum Probabity to filter")
args = vars(ap.parse_args())

print("[INFO] loading face detector model...")
prototxt_path = os.path.sep.join([args["face"], "deploy.prototxt"])
weights_path = os.path.sep.join([args["face"], "res10_300x300_ssd_iter_140000.caffemodel"])
face_net = cv2.dnn.readNet(prototxt_path, weights_path)

print("[INFO] loading face mask detector model...")
mask_net = load_model(args["model"])

print("[INFO] starting video stream...")
vs = VideoStream(src = 0).start()
time.sleep(2.0)

while True:

    frame = vs.read()
    frame = imutils.resize(frame, width = 400)

    (locs, prediction) = detect_and_predict(frame, face_net, mask_net)

    for (box, pred) in zip(locs, prediction):

        (start_x, start_y, end_x, end_y) = box
        (mask, without_mask) = pred

        label = "Mask" if mask > without_mask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        label = "{}: {:.2f}%".format(label, max(mask, without_mask) * 100)

        cv2.putText(frame, label, (start_x, start_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), color, 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

cv2.destroyAllWindows()
vs.stop()

#import
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from imutils import paths

#argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required = True, help = "Input Dataset Path")
ap.add_argument("-p", "--plot", type = str, default = "plot.png", help = "Output Path")
ap.add_argument("-m", "--model", type = str, default = "mask_detector.model", help = "Model Path")
args = vars(ap.parse_args())

#Parameters
INIT_LR = 1e-4
EPOCHS = 20
BS = 32

#Pre-processing Data
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
data = []
labels = []

for imagePath in imagePaths:

    #Set Label
    label = imagePath.split(os.path.sep)[-2] #without_mask or with_mask

    #Load Image
    image = load_img(imagePath, target_size = (224, 224))
    image = img_to_array(image)
    image = preprocess_input(image)

    data.append(image)
    labels.append(label)

#Convert to numpy
data = np.array(data, dtype = "float32")
labels = np.array(labels)

#One Hot Encoding
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

#Train and Test Data
(train_x, test_x, train_y, test_y) = train_test_split(data, labels, test_size = 0.20, stratify = labels, random_state = 42)

#Data Augmentation
aug = ImageDataGenerator(rotation_range = 20, zoom_range = 0.15, width_shift_range = 0.2, height_shift_range = 0.2, shear_range = 0.15, horizontal_flip = True, fill_mode = "nearest")

#MobilenetV2 Model
base_model = MobileNetV2(weights = "imagenet", include_top = False, input_tensor = Input(shape = (224, 224, 3)))

#Head
head_model = base_model.output
head_model = AveragePooling2D(pool_size = (7, 7))(head_model)
head_model = Flatten(name = "flatten")(head_model)
head_model = Dense(128, activation = "relu")(head_model)
head_model = Dropout(0.5)(head_model)
head_model = Dense(2, activation = "softmax")(head_model)

#model
model = Model(inputs = base_model.input, outputs = head_model)

for layer in base_model.layers:
    layer.trainable = False

#Compile model
print("[INFO] compiling model...")
opt = Adam(lr = INIT_LR, decay = INIT_LR / EPOCHS)
model.compile(loss = "binary_crossentropy", optimizer = opt, metrics = ["accuracy"])

#Train model
print("[INFO] training head...")
H = model.fit(
    aug.flow(train_x, train_y, batch_size = BS),
    steps_per_epoch = len(train_x) // BS,
    validation_data = (test_x, test_y),
    validation_steps = len(test_x) // BS,
    epochs = EPOCHS
)

#Prediction
print("[INFO] evaluating network...")
pred_index = model.predict(test_x, batch_size = BS)

pred_index = np.argmax(pred_index, axis = 1)

#classification_report
print(classification_report(test_y.argmax(axis = 1), pred_index, target_names = lb.classes_))

#Save model
print("[INFO] saving mask detector model...")
model.save(args["model"], save_format = "h5")

#Plot Training loss
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label = "train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label = "val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label = "train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label = "val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc = "lower left")
plt.savefig(args["plot"])

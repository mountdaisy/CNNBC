import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import MultiLabelBinarizer
#from sklearn.preprocessing import Binarizer
from sklearn.model_selection import train_test_split
#from kc.VGGArch import VGGArch
from kc.XceptionArch import XceptionArch
#from kc.InceptionResNetV2Arch import InceptionResNetV2Arch
#from kc.InceptionV3Arch import InceptionV3Arch
#from kc.NASNetMobileArch import NASNetMobileArch
import matplotlib.pyplot as plt
from imutils import paths
import ntpath
import numpy as np
import random
import pickle
import cv2
import os
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils.training_utils import multi_gpu_model
import tensorflow as tf

architecture ="XceptionArch" # VGGArch,XceptionArch,InceptionResNetV2Arch,InceptionV3Arch,NASNetMobileArch
patchType="Binary"# Binary,Master
IMAGE_SIZE=299 #200
BS = 64
#BS=32 for DenseNet

modelToUse = architecture+"-1-"+patchType+"-"+str(IMAGE_SIZE)
pathToDataset ="C:\\Project\\Patches\\Train\\"+patchType+"\\"
pathToModel=".\\output\\"+modelToUse+".model"
pathToLabel=".\\output\\"+modelToUse+".pickle"
pathToTrainLossImage =".\\output\\"+modelToUse+".png"

GPUS=1
EPOCHS = 100
PATIENCE=20
INIT_LR = 1e-3
IMAGE_DIMS = (IMAGE_SIZE, IMAGE_SIZE, 3)

print("[INFO] loading images...")
imagePaths = sorted(list(paths.list_images(pathToDataset)))
random.seed(42)
random.shuffle(imagePaths)

data = []
labels = []

for imagePath in imagePaths:
	image = cv2.imread(imagePath)
	image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
	image = img_to_array(image)
	data.append(image)

	fileName = ntpath.basename(imagePath).split('.')[0]
	l = label = [fileName.split('-')[4]]
	labels.append(l)

data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

print("[INFO] data matrix: {} images ({:.2f}MB)".format(
	len(imagePaths), data.nbytes / (1024 * 1000.0)))

print("[INFO] class labels:")
mlb = MultiLabelBinarizer()
labels = mlb.fit_transform(labels)

for (i, label) in enumerate(mlb.classes_):
	print("{}. {}".format(i + 1, label))

(trainX, testX, trainY, testY) = train_test_split(data,
	labels, test_size=0.2, random_state=42)

aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")

print("[INFO] compiling model {} ...".format(architecture))

if architecture=="VGGArch":
	model = VGGArch.build(
		width=IMAGE_DIMS[1], height=IMAGE_DIMS[0],
		depth=IMAGE_DIMS[2], classes=len(mlb.classes_),
		finalAct="sigmoid")
elif architecture=="XceptionArch":
	model = XceptionArch.build(
		width=IMAGE_DIMS[1], height=IMAGE_DIMS[0],
		depth=IMAGE_DIMS[2], classes=len(mlb.classes_),
		finalAct="sigmoid")
elif architecture=="InceptionResNetV2Arch":
	model = InceptionResNetV2Arch.build(
		width=IMAGE_DIMS[1], height=IMAGE_DIMS[0],
		depth=IMAGE_DIMS[2], classes=len(mlb.classes_),
		finalAct="sigmoid")
elif architecture=="InceptionV3Arch":
	model = InceptionV3Arch.build(
		width=IMAGE_DIMS[1], height=IMAGE_DIMS[0],
		depth=IMAGE_DIMS[2], classes=len(mlb.classes_),
		finalAct="sigmoid")
elif architecture=="NASNetMobileArch":
	model = NASNetMobileArch.build(
		width=IMAGE_DIMS[1], height=IMAGE_DIMS[0],
		depth=IMAGE_DIMS[2], classes=len(mlb.classes_),
		finalAct="sigmoid")

opt = SGD(lr=INIT_LR, momentum=0.9)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

checkpoint = ModelCheckpoint(pathToModel, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=PATIENCE, verbose=1, mode='auto')
print("[INFO] training network...")
H = model.fit_generator(
	aug.flow(trainX, trainY, batch_size=BS),
	validation_data=(testX, testY),
	steps_per_epoch=len(trainX) // BS,
	epochs=EPOCHS, verbose=1,callbacks=[checkpoint,early])

print("[INFO] serializing label binarizer...")
f = open(pathToLabel, "wb")
f.write(pickle.dumps(mlb))
f.close()

N = 0
if early.stopped_epoch < EPOCHS and early.stopped_epoch >0:
	N=early.stopped_epoch+1
else:
	N=EPOCHS

print("N {}".format(N))

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper left")
plt.savefig(pathToTrainLossImage)

from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os
import glob
import ntpath
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns


architecture ="XceptionArch" # VGGArch,XceptionArch,InceptionResNetV2Arch,InceptionV3Arch,NASNetMobileArch
patchType="Binary"# Binary,Master
IMAGE_SIZE=299 #299


modelToUse = architecture+"-1-"+patchType+"-"+str(IMAGE_SIZE)
modelPath="C:\\Project\\output\\"+modelToUse+".model"
labelPath="C:\\Project\\output\\"+modelToUse+".pickle"
imagePath="C:\\Project\\Patches\\Test\\"+patchType+"\\"
pathToCMImage =".\\output\\"+modelToUse+"-confusionMatrix2.png"

print("[INFO] loading network...")
model = load_model(modelPath)
mlb = pickle.loads(open(labelPath, "rb").read())
correct=0
wrong=0

actual = []
predicated = []
labels = []
if patchType == 'Master':
	labels = ['healthy', 'low', 'medium', 'high'] 
else:
	labels = ['cancer', 'healthy'] 

imageToReview = glob.glob(str(imagePath)+r'\*')
for img in imageToReview:
	image = cv2.imread(img)
	fileName = ntpath.basename(img).split('.')[0]
	knownLabel = fileName.split('-')[4]
	print(fileName)
	image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
	image = image.astype("float") / 255.0
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)

	proba = model.predict(image)[0]
	idxs = np.argsort(proba)[::-1][:1]

	for (i, j) in enumerate(idxs):
		label = "{}: {:.2f}%".format(mlb.classes_[j], proba[j] * 100)
		actual.append(knownLabel) 
		predicated.append(mlb.classes_[j]) 
		print ("Known: {} Guess: {}".format(knownLabel,label))
		if knownLabel==mlb.classes_[j]:
			correct = correct+1
		else:
			wrong =wrong+1

print ("Wrong classificiation {}".format(wrong))
print ("Correct classificiation {}".format(correct))
print ("Success rate: {:.2f}%".format(correct/(correct+wrong) * 100))

cm = confusion_matrix(actual, predicated, labels) 
sns.heatmap(cm, annot=True, 
            fmt='d', cmap='PuBuGn'
			,xticklabels=labels, yticklabels=labels)
plt.title('Confusion matrix - '+architecture+' '+patchType) 
plt.xlabel('Predicted') 
plt.ylabel('True') 
plt.savefig(pathToCMImage)
	# show the output image
	#cv2.imshow("Output", output)
	#cv2.waitKey(0)

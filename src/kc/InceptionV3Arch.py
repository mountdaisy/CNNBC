from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Input,GlobalAveragePooling2D
from keras.models import Model

class InceptionV3Arch:
	@staticmethod
	def build(width, height, depth, classes, finalAct="softmax"):
		input_tensor = Input(shape=(width, height, depth))
		base_model = InceptionV3(input_tensor=input_tensor, weights=None, include_top=False)
		x = base_model.output
		x = GlobalAveragePooling2D()(x)
		x = Dense(1024, activation='relu')(x)
		predictions = Dense(classes, activation=finalAct)(x)
		model = Model(inputs=base_model.input, outputs=predictions)
		model.summary()
		return model
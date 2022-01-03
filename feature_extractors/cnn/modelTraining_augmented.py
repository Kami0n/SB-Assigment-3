import sys
sys.path.append(sys.path[0]+'\\..\\..')

import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical

import numpy as np
import cv2
import random
import json
import glob

from preprocessing.preprocess import Preprocess

from commonFunctions import get_annotations

def modelArchitecture(IMG_WIDTH, IMG_HEIGHT): # Create the model
	model = Sequential()
	model.add(Conv2D(64, kernel_size=4, strides=1,activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 1)))
	model.add(Conv2D(64, kernel_size=4, strides=2,activation='relu'))
	model.add(MaxPooling2D(pool_size=2))
	model.add(Conv2D(128, kernel_size=4, strides=1,activation='relu'))
	model.add(MaxPooling2D(pool_size=2))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(101, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) # Compile the model
	return model

def main():
	resize = 100
	# small size images (not more than 200 * 200)
	IMG_WIDTH = resize
	IMG_HEIGHT = resize
	training_data = []

	print("Training Data")
	
	X_train = []
	Y_train = []
	
	for class_num in range(1,101):
		pathAugmented = 'data/perfectly_detected_ears/trainClassedAugmented/'+str(class_num)+'/'
		
		for img in os.listdir(pathAugmented):
			try:
				img_array = cv2.imread(os.path.join(pathAugmented, img), cv2.IMREAD_GRAYSCALE)
				new_array = cv2.resize(img_array, (IMG_WIDTH, IMG_HEIGHT))
				training_data.append([new_array, class_num])
			except Exception as e:
				pass
	
	np.random.seed(0)
	random.shuffle(training_data) # shuffle for better training and learning of the machine
	
	X_train = [] # feature set
	Y_train = [] # labels
	
	for features, labels in training_data:
		X_train.append(features)
		Y_train.append(labels)
	
	X_train = np.array(X_train).reshape(-1, IMG_WIDTH, IMG_HEIGHT, 1)
	X_train = X_train/255.0
	
	model = modelArchitecture(IMG_WIDTH, IMG_HEIGHT)
	# Train the model
	model.fit(X_train, to_categorical(Y_train), batch_size=32, epochs=100)
	# increase the epochs or decrease the batch size according to classes
	model.save('model_3_augmented.h5')

if __name__ == '__main__':
	main()
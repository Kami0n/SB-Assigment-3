import cv2
import numpy as np

import tensorflow as tf
from tensorflow.keras import models

class CNN:
	def __init__(self, resize=100):
		self.resize=resize
		self.model = models.load_model('model_1.h5')
		# model_1
		# model_2_augmented
		
	def predict(self, img):
		img_array = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		new_array = cv2.resize(img_array, (self.resize, self.resize))  # make all images of standard/same size
		X_predict = np.array(new_array).reshape(-1, self.resize, self.resize, 1)
		X_predict = X_predict/255.0
		
		prediction = self.model.predict(X_predict)
		predictionThis = prediction[0]
		
		best_class = np.argmax(predictionThis)
		return predictionThis, best_class

if __name__ == '__main__':
	cnn_predicotor = CNN()
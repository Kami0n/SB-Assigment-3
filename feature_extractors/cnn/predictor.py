import cv2
import numpy as np

import tensorflow as tf
from tensorflow.keras import models

class CNN:
	def __init__(self, resize=100):
		self.resize=resize
		self.model = models.load_model('model_opt.h5') #

	def predict(self, img):
		img_array = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		new_array = cv2.resize(img_array, (self.resize, self.resize))  # make all images of standard/same size
		X_predict = np.array(new_array).reshape(-1, self.resize, self.resize, 1)
		
		prediction = self.model.predict(X_predict)
		
		best_class = np.argmax(prediction[0]) 
		return prediction, best_class

if __name__ == '__main__':
	cnn_predicotor = CNN()
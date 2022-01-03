import sys
sys.path.append(sys.path[0]+'\\..\\..')
import os
import numpy as np
import cv2
import random
import json
import glob
from preprocessing.preprocess import Preprocess
from commonFunctions import get_annotations

import torch
from torchvision.models.resnet import resnet18

def main():
	resize = 100
	with open('config_recognition.json') as config_file:
		config = json.load(config_file)
	images_path = config['train_images_path']
	annotations_path = config['annotations_path']
	
	im_list = sorted(glob.glob(images_path + '/*.png', recursive=True))
	im_list = [i.replace('\\', '/') for i in im_list] # windows backslash weirdness fix
	
	cla_d = get_annotations(annotations_path)
	
	# small size images (not more than 200 * 200)
	IMG_WIDTH = resize
	IMG_HEIGHT = resize
	training_data = []

	print("Training Data")
	
	X_train = []
	Y_train = []
	
	for im_name in im_list:
		try:
			img_array = cv2.imread(im_name, cv2.IMREAD_GRAYSCALE) # convert in gray scale for faster computation
			new_array = cv2.resize(img_array, (IMG_WIDTH, IMG_HEIGHT))  # make all images of standard/same size
			real_class = cla_d['/'.join(im_name.split('/')[-2:])]
			training_data.append([new_array, real_class])    # assign class to image
		except Exception as e:
			pass
	
	np.random.seed(0)
	random.shuffle(training_data)   # shuffle for better training and learning of the machine
	
	X_train = [] # feature set
	Y_train = [] # labels
	
	for features, labels in training_data:
		X_train.append(features)
		Y_train.append(labels)
	
	X_train = np.array(X_train).reshape(-1, IMG_WIDTH, IMG_HEIGHT, 1)
	X_train = X_train/255.0
	
	
	
	

if __name__ == '__main__':
	main()
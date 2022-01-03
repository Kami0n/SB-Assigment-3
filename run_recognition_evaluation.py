import cv2
import numpy as np
np.set_printoptions(suppress=True)
import glob
import os
import json
from pathlib import Path
from scipy.spatial.distance import cdist

from preprocessing.preprocess import Preprocess
from metrics.evaluation_recognition import Evaluation

from commonFunctions import get_annotations

class EvaluateAll:

	def __init__(self):
		os.chdir(os.path.dirname(os.path.realpath(__file__)))
		
		with open('config_recognition.json') as config_file:
			config = json.load(config_file)
		
		self.images_path = config['images_path']
		self.annotations_path = config['annotations_path']
	
	def run_evaluation(self):
		im_list = sorted(glob.glob(self.images_path + '/*.png', recursive=True))
		im_list = [i.replace('\\', '/') for i in im_list] # windows backslash weirdness fix
		preprocObj = Preprocess()
		eval = Evaluation()
		cla_d = get_annotations(self.annotations_path)
		y = [] # real classes, real identity class
		
		# False True
		pix = False
		lbpShow = False
		cnn = True
		
		# Change the following extractors, modify and add your own
		if pix: # Pixel-wise comparison:
			import feature_extractors.pix2pix.extractor as p2p_ext
			pix2pix = p2p_ext.Pix2Pix()
			plain_features_arr = []
		
		if lbpShow: # LBP
			import feature_extractors.lbp.extractor as lbp_ext
			lbp = lbp_ext.LBP()
			lbp_features_arr = []
		
		if cnn: # CNN
			import feature_extractors.cnn.predictor as cnn_pred
			# model_1
			# model_2_augmented
			# model_3_augmented
			cnn = cnn_pred.CNN(predictor='model_1.h5')
		
		class_array = []
		scores_array = []
		
		for im_name in im_list:
			# Read an image
			img = cv2.imread(im_name)
			print('/'.join(im_name.split('/')[-2:]))
			y.append(cla_d['/'.join(im_name.split('/')[-2:])])
			
			# Apply some preprocessing here
			#img = preprocObj.histogram_equlization_rgb(img)
			
			# Run the feature extractors
			if pix:
				plain_features = pix2pix.extract(img)
				plain_features_arr.append(plain_features)
			
			if lbpShow:
				lbp_features = lbp.extract(img)
				lbp_features_arr.append(lbp_features)
				#lbp_scores, best_class = lbp.predict(img)
				#scores_array.append(lbp_scores)
				#class_array.append(best_class)
			
			if cnn:
				scores, best_class = cnn.predict(img)
				class_array.append(best_class)
				scores_array.append(np.array(scores).reshape(1,-1))
		
		# calculate the distance between images
		if pix:
			Y_plain_pix = cdist(plain_features_arr, plain_features_arr, 'jensenshannon')
			r1_pix = eval.compute_rank1(Y_plain_pix, y)
			print('\nPix2Pix Rank-1 [%] ', round(r1_pix,2))
		
		if lbpShow:
			Y_LBP = cdist(lbp_features_arr, lbp_features_arr, 'jensenshannon')
			r1_LBP = eval.compute_rank1(Y_LBP, y)
			print('\nLBP Rank-1 [%] ', round(r1_LBP,2))
		
		if cnn:
			#percent = eval.computeCorrectClasses(class_array,y)
			#print('\nCNN Percent [%] ', percent)
			
			r1_CNN = eval.compute_rank1_accuracy(scores_array, y)
			print('\nCNN Rank-1 [%] ', r1_CNN)
			
			r5_CNN = eval.compute_rank5_accuracy(scores_array, y)
			print('\nCNN Rank-5 [%] ', r5_CNN)
			
			eval.plotCMC(scores_array, y)
			

if __name__ == '__main__':
	ev = EvaluateAll()
	ev.run_evaluation()
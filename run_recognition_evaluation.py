import cv2
import numpy as np
import glob
import os
import json
from pathlib import Path
from scipy.spatial.distance import cdist

from preprocessing.preprocess import Preprocess
from metrics.evaluation_recognition import Evaluation

class EvaluateAll:

	def __init__(self):
		os.chdir(os.path.dirname(os.path.realpath(__file__)))
		
		with open('config_recognition.json') as config_file:
			config = json.load(config_file)
		
		self.images_path = config['images_path']
		self.annotations_path = config['annotations_path']
	
	def clean_file_name(self, fname):
		return fname.split('/')[1].split(' ')[0]
	
	def get_annotations(self, annot_f):
		d = {}
		with open(annot_f) as f:
			lines = f.readlines()
			for line in lines:
				(key, val) = line.split(',')
				# keynum = int(self.clean_file_name(key))
				d[key] = int(val)
		return d
	
	def run_evaluation(self):
		im_list = sorted(glob.glob(self.images_path + '/*.png', recursive=True))
		im_list = [i.replace('\\', '/') for i in im_list] # windows backslash weirdness fix
		#iou_arr = []
		preprocObj = Preprocess()
		eval = Evaluation()
		cla_d = self.get_annotations(self.annotations_path)
		
		# Change the following extractors, modify and add your own
		# Pixel-wise comparison:
		import feature_extractors.pix2pix.extractor as p2p_ext
		pix2pix = p2p_ext.Pix2Pix()
		
		# LBP
		import feature_extractors.lbp.extractor as lbp_ext
		lbp = lbp_ext.LBP()
		
		plain_features_arr = []
		
		lbp_features_arr = []
		lbp_scores_arr = []
		lbp_classes_arr = []
		y = [] # real classes, real identity class
		
		for im_name in im_list:
			# Read an image
			img = cv2.imread(im_name)
			print('/'.join(im_name.split('/')[-2:]))
			y.append(cla_d['/'.join(im_name.split('/')[-2:])])
			
			# Apply some preprocessing here
			img = preprocObj.histogram_equlization_rgb(img)
			
			# Run the feature extractors
			#plain_features = pix2pix.extract(img)
			#plain_features_arr.append(plain_features)
			
			#lbp_features = lbp.extract(img)
			#lbp_features_arr.append(lbp_features)
			
			lbp_scores, best_class = lbp.predict(img)
			lbp_scores_arr.append(lbp_scores)
			lbp_classes_arr.append(best_class)
			
		#Y_plain_pix = cdist(plain_features_arr, plain_features_arr, 'jensenshannon')
		#r1_pix = eval.compute_rank1(Y_plain_pix, y)
		#print('Pix2Pix Rank-1 [%] ', r1_pix)
		
		percent = eval.computeCorrectClasses(lbp_classes_arr,y)
		print('LBP percent [%] ', percent)
		
		lbp_scores_arr = np.array(lbp_scores_arr)
		r1_LBP = eval.compute_rank1(lbp_scores_arr, y)
		print('LBP Rank-1 [%] ', r1_LBP)
		

if __name__ == '__main__':
	ev = EvaluateAll()
	ev.run_evaluation()
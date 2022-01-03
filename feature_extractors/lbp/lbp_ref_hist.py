import sys
sys.path.append(sys.path[0]+'\\..\\..')

import cv2
import glob
import json
import pandas as pd

import feature_extractors.lbp.extractor as lbp_ext
from preprocessing.preprocess import Preprocess

# calculate reference histograms from training data
def main():
	with open('config_recognition.json') as config_file:
		config = json.load(config_file)
	images_path = config['train_images_path']
	annotations_path = config['annotations_path']
	
	im_list = sorted(glob.glob(images_path + '/*.png', recursive=True))
	im_list = [i.replace('\\', '/') for i in im_list] # windows backslash weirdness fix
	
	cla_d = get_annotations(annotations_path)
	preprocObj = Preprocess()
	lbp = lbp_ext.LBP()
	
	#lbp_features_arr = []
	y = [] # real classes, real identity class
	referenceData = []
	
	for im_name in im_list:
		# Read an image
		img = cv2.imread(im_name)
		print('/'.join(im_name.split('/')[-2:]))
		real_class = cla_d['/'.join(im_name.split('/')[-2:])]
		y.append(real_class)
		
		# Apply some preprocessing here
		img = preprocObj.histogram_equlization_rgb(img)
		
		lbp_features = lbp.extract(img)
		#lbp_features_arr.append(lbp_features)
		
		referenceData.append([im_name, real_class, lbp_features])
	
	referenceDataDf = pd.DataFrame(data=referenceData, columns=['imagePath','class','histogram'])
	referenceDataDf.to_pickle('data/perfectly_detected_ears/pickle/reference_hist.pkl')
	print(referenceDataDf)

def get_annotations(annot_f):
	d = {}
	with open(annot_f) as f:
		lines = f.readlines()
		for line in lines:
			(key, val) = line.split(',')
			# keynum = int(self.clean_file_name(key))
			d[key] = int(val)
	return d

def clean_file_name(fname):
	return fname.split('/')[1].split(' ')[0]

if __name__ == '__main__':
	main()
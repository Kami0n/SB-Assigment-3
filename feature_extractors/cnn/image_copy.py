import sys
sys.path.append(sys.path[0]+'\\..\\..')

import os
import json
import glob
import shutil

from commonFunctions import get_annotations

def main():
	with open('config_recognition.json') as config_file:
		config = json.load(config_file)
	images_path = config['train_images_path']
	annotations_path = config['annotations_path']
	
	im_list = sorted(glob.glob(images_path + '/*.png', recursive=True))
	im_list = [i.replace('\\', '/') for i in im_list] # windows backslash weirdness fix
	
	cla_d = get_annotations(annotations_path)
	for im_name in im_list:
		real_class = cla_d['/'.join(im_name.split('/')[-2:])]
		pathDir = 'data/perfectly_detected_ears/trainClassed/'+str(real_class)
		if not os.path.exists(pathDir):
			os.makedirs(pathDir) 
		
		original = im_name
		target = pathDir+'/'+'/'.join(im_name.split('/')[-1:])
		shutil.copyfile(original, target)
		

if __name__ == '__main__':
	main()
import sys
sys.path.append(sys.path[0]+'\\..\\..')

import os
from os import walk
import cv2
import random
import numpy as np
import shutil
def brightness(img, low, high):
	value = random.uniform(low, high)
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	hsv = np.array(hsv, dtype = np.float64)
	hsv[:,:,1] = hsv[:,:,1]*value
	hsv[:,:,1][hsv[:,:,1]>255]  = 255
	hsv[:,:,2] = hsv[:,:,2]*value 
	hsv[:,:,2][hsv[:,:,2]>255]  = 255
	hsv = np.array(hsv, dtype = np.uint8)
	img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
	return img

def channel_shift(img, value):
	value = int(random.uniform(-value, value))
	img = img + value
	img[:,:,:][img[:,:,:]>255]  = 255
	img[:,:,:][img[:,:,:]<0]  = 0
	img = img.astype(np.uint8)
	return img

def horizontal_flip(img, flag):
	if flag:
		return cv2.flip(img, 1)
	else:
		return img

def rotate_image(image, angle):
	image_center = tuple(np.array(image.shape[1::-1]) / 2)
	rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
	result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
	return result

def adjust_gamma(image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)

def main():
	# gres po vseh razredih (mapah)
	for classNumber in range(1, 101):
		
		pathClass= 'data/perfectly_detected_ears/trainClassed/'+str(classNumber)+'/'
		filenames = next(walk(pathClass), (None, None, []))[2]  # [] if no file
		# gres po vseh slikah v mapi
		for imageName in filenames:
			print(imageName)
			
			pathAugmented = 'data/perfectly_detected_ears/trainClassedAugmented/'+str(classNumber)+'/'
			if not os.path.exists(pathAugmented):
				os.makedirs(pathAugmented) 
			
			original = pathClass+imageName
			target = pathAugmented+imageName
			shutil.copyfile(original, target)
			
			imgOrg = cv2.imread(target)
			name = imageName.split('.')[0]
			i = 1
			
			# brightness
			img1 = brightness(imgOrg, 0.5, 1.5)
			cv2.imwrite(pathAugmented+name+'_'+str(i)+'_.png', img1)
			i+=1
			
			# Channel shifting
			#img = channel_shift(imgOrg, 60)
			#cv2.imwrite(pathAugmented+name+'_'+str(i)+'_.png', img)
			#i+=1
			
			# Gammma adjust
			for gamma in [1.5, 0.5]:
				adjusted = adjust_gamma(imgOrg, gamma=gamma)
				cv2.imwrite(pathAugmented+name+'_'+str(i)+'_.png', adjusted)
				i+=1
			
			# Horizontal flipping
			img2 = horizontal_flip(imgOrg, True) 
			cv2.imwrite(pathAugmented+name+'_'+str(i)+'_.png', img2)
			i+=1
			
			# Scaling by a factor in the range (0.9, 1.2)
			
			
			# Blurring with Gaussian filters with s in the range (0, 0.5)
			for blur in [3, 5, 9, 13, 21]:
				blurImg = cv2.GaussianBlur(imgOrg,(blur,blur),cv2.BORDER_DEFAULT)
				cv2.imwrite(pathAugmented+name+'_'+str(i)+'_.png', blurImg)
				i+=1
			
			# Rotations up to 35
			for angle in range(-35, 35, 5):
				if( angle > -10 and angle < 10 ):
					continue
				img3 = rotate_image(imgOrg, angle)
				cv2.imwrite(pathAugmented+name+'_'+str(i)+'_.png', img3)
				i+=1
			
			# nove slike se shrani v trainClassedAugmented

if __name__ == '__main__':
	main()
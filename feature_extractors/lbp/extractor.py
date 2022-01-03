import cv2, sys
from skimage import feature
import numpy as np
import pandas as pd

class LBP:
	def __init__(self, num_points=8, radius=2, eps=1e-7, resize=100):
		self.num_points = num_points * radius
		self.radius = radius
		self.resize = resize
		self.eps = eps
		#self.referenceDataDf = pd.read_pickle('data/perfectly_detected_ears/pickle/reference_hist.pkl')# 'imagePath','class','histogram'
	
	def extract(self, img):
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		img = cv2.resize(img, (self.resize, self.resize))
		#cv2.imshow("image", img)
		
		lbp = feature.local_binary_pattern(img, self.num_points, self.radius, method="uniform")
		#cv2.imshow("lbp", lbp)
		#cv2.waitKey(0)
		
		n_bins = int(lbp.max() + 1)
		#hist, _ = np.histogram(lbp.ravel(), density=True, bins=n_bins, range=(0, n_bins))
		hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, self.num_points + 3), range=(0, self.num_points + 2)) # density=True, 
		
		hist = hist.astype("float")
		hist /= (hist.sum() + self.eps)
		return hist
		#return lbp.ravel()
	
	def predict(self, img):
		hist = self.extract(img)
		best_score = 10
		best_class = None
		scores = []
		for index, row in self.referenceDataDf.iterrows():
			score = self.kullback_leibler_divergence(hist, row['histogram'])
			scores.append(score)
			if score < best_score:
				best_score = score
				best_class = row['class']
		return scores, best_class
	
	def kullback_leibler_divergence(self, p, q):
		p = np.asarray(p)
		q = np.asarray(q)
		filt = np.logical_and(p != 0, q != 0)
		return np.sum(p[filt] * np.log2(p[filt] / q[filt]))

if __name__ == '__main__':
	fname = sys.argv[1]
	img = cv2.imread(fname)
	extractor = LBP()
	features = extractor.extract(img)
	print(features)
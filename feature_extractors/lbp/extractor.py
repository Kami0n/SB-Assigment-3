import cv2, sys
from skimage import feature
import numpy as np
import pandas as pd

class LBP:
	def __init__(self, num_points=10, radius=8, eps=1e-6, resize=100):
		self.num_points = num_points * radius
		self.radius = radius
		self.resize = resize
		self.eps = eps
		
		# 'imagePath','class','histogram'
		self.referenceDataDf = pd.read_pickle('data/perfectly_detected_ears/pickle/reference_hist.pkl')
	
	def extract(self, img):
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		img = cv2.resize(img, (self.resize, self.resize))
		
		lbp = feature.local_binary_pattern(img, self.num_points, self.radius, method="uniform")
		
		n_bins = int(lbp.max() + 1)
		hist, _ = np.histogram(lbp, density=True, bins=n_bins, range=(0, n_bins))
		
		hist = hist.astype("float")
		hist /= (hist.sum() + self.eps)
		
		#cv2.imshow("image", img)
		#cv2.waitKey(0)
		#cv2.imshow("lbp", lbp)
		#cv2.waitKey(0)
		
		return hist
	
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
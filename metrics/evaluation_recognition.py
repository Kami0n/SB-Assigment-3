import math
import numpy as np

class Evaluation:
	def compute_rank1(self, Y, y):
		classes = np.unique(sorted(y))
		count_all = 0
		count_correct = 0
		
		for cla1 in classes: # gre skozi vse razrede, ki so v annotations
			idx1 = y==cla1 # nastavi true v tabeli vse primere, ki so razred cla1
			
			# Compute only for cases where there is more than one sample:
			if (list(idx1).count(True)) <= 1:
				continue
			
			Y1 = Y[idx1==True, :]
			Y1[Y1==0] = math.inf # same image => infinity
			
			for y1 in Y1:
				s = np.argsort(y1)
				smin = s[0] # get closest ear
				imin = idx1[smin] # get if true class
				count_all += 1
				if imin:
					count_correct += 1
		
		return count_correct/count_all*100
	
	# Add your own metrics here, such as rank5, (all ranks), CMC plot, ROC, ...
	
	def compute_rank5(self, Y, y):
		classes = np.unique(sorted(y))
		sentinel = 0
		for cla1 in classes:
			# First loop over classes in order to select the closest for each class.
			idx1 = y==cla1
			if (list(idx1).count(True)) <= 1:
				continue
			Y1 = Y[idx1==True, :]
			Y1[Y1==0] = math.inf
			
			for cla2 in classes:
				# Select the closest that is higher than zero:
				idx2 = y==cla2
				if (list(idx2).count(True)) <= 1:
					continue
				Y2 = Y1[:, idx1==True]
				Y2[Y2==0] = math.inf
				min_val = np.min(np.array(Y2))
				# ...
	
	def CMCplot():
		pass
	
	def ROCplot():
		pass
	
	def computeCorrectClasses(self, predY, trueY):
		correct = 0
		for indx, item in enumerate(predY):
			if(trueY[indx] == item):
				correct += 1
		percent = round((correct/len(predY))*100, 2)
		return percent
import numpy as np 
import matplotlib.pyplot as plt
import code

def plotHist(X, bins = 50):
	# code.interact(local=dict(globals(), **locals()))
	X = X[X >= 1000000]
	vals = X.flatten()
	vals_no = np.setdiff1d(vals,0)
	vals_no.shape
	plt.hist(vals_no, bins)
	plt.show()
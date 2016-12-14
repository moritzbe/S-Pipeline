from data_tools import *
from plotLib import *
from algorithms import *
from plot_lib import *
from nets import *
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.feature_selection import RFE
import numpy as np
import matplotlib.pyplot as plt
import code 

# Loading the data into X, y
Data = loadnumpy("../full_ratingsdata.npy")



plotHist(Data[:,1:3], bins = 50)

# Column of revenue 2015
label_column = 3

X = np.delete(Data[:,1:], np.s_[label_column-1], 1)

# multiply by 10000, so small values between 0 and 1 are brought to the scale of revenues.
# This is just arbitrary. Could also normalize revenue (devide by max(revenue))
X[:,label_column:] = X[:,label_column:]*10000
X = X[:,label_column-1:]

# In this case, X_test is the data used to predict 2016. Bad Naming!
# exclude earliest timestep, include 2015 to predict 2016
X_test = Data[:,label_column-1:] 
X_test[:,label_column:] = X_test[:,label_column:]*10000
X_test = X_test[:,label_column-1:]

# Labels for learning (2015 revenues)
y = Data[:,label_column]

# Number of entries
m = X.shape[0]

print "The number of training samples m is", m

# # Cross Validation
cv = 4


################## Linear Regression (Ridge) #################
X_off = addOffset(X) # Add offset (ones) for lin. regression
X_test_off = addOffset(X_test)
lin_reg = linRegress(X_off, y)
scores = np.mean(cross_val_score(lin_reg, X_off, y, scoring='neg_mean_absolute_error', cv=cv))
print "The train accuracy of Linear Regression is", scores


#######    Random Forests #########
# Purchase Prediction
# Revenues < 1000 are considered 0
y[y<1000] = 0
y[y>=1000] = 1 #if revenue is expected, y = 1
K = 30 # Number of trees. Computing cost/time scales exponentially with K
rf = randForest(X[::10], y[::10], K)
print "The train accuracy of Random Forest is", np.mean(cross_val_score(rf, X, y, cv=cv))
y_zeros = rf.predict(X_test)


# Predict:
y_pred = lin_reg.predict(X_test_off)
y_pred[y_pred < 0] = 0
# can be used with or without purchase prediction
total_error = np.sum(np.abs(y_pred*y_zeros - y))
mean_error = np.mean(np.abs(y_pred*y_zeros - y))

print "TOTAL ERROR IS", total_error
print "MEAN ERROR IS", mean_error


################## SVM #######################################
# y[y<10000/rev_maximum] = 0
# y[y>=10000/rev_maximum] = 1
# X_off = addOffset(X)
# svm = supvecm(X_off, y, 5)
# train_accuracy = np.mean(cross_val_score(svm, X_off, y, cv=cv))
# print "The train accuracy of SVM is", train_accuracy
# y_zeros = svm.predict(X_off)

# total_error = np.sum(np.abs((y_pred*y_zeros - y)))
# print total_error/m

#############  Most simple submission ########################
# y_pred = y2014
# y2 = Data[:,-2]
# a = np.abs(y2-y)
# print np.mean(a)
# plotConfusionMatrix(y, y_pred)

# use for debugging:
# code.interact(local=dict(globals(), **locals()))

## Evaluation
difference = np.sum(np.absolute(y_pred*y_zeros - y)) 
print difference

# Submission:
saveToCSV(Data, y_pred * y_zeros, "final2")

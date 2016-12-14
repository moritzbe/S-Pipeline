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
# Data = loadnumpy("../full_data.npy")

Data = loadnumpy("../full_ratingsdata.npy")
#  rest_ids = loadnumpy("ids_data.npy")



plotHist(Data[:,1:3], bins = 50)

# # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
label_column = 3
# rev_maximum = np.max(Data[:,1:label_column-1])

X = np.delete(Data[:,1:], np.s_[label_column-1], 1)
X[:,label_column:] = X[:,label_column:]*10000
X = X[:,label_column-1:]
X_test = Data[:,2:] 
X_test[:,label_column:] = X_test[:,label_column:]*10000
X_test = X_test[:,label_column-1:]

y = Data[:,label_column]


m = X.shape[0]

print "The number of training samples m is", m

# # Cross Validation
cv = 4


################## Linear Regression (Ridge) #################
X_off = addOffset(X)
lin_reg = linRegress(X_off, y)
scores = np.mean(cross_val_score(lin_reg, X_off, y, scoring='neg_mean_absolute_error', cv=cv))
print "The train accuracy of Linear Regression is", scores
X_test_off = addOffset(X_test)


#######    Random Forests #########
y[y<1000] = 0
y[y>=1000] = 1
K = 30
rf = randForest(X[::10], y[::10], K)
print "The train accuracy of Random Forest is", np.mean(cross_val_score(rf, X, y, cv=cv))
y_zeros = rf.predict(X_test)


# Predict:
# y_pred = lin_reg.predict(X_test_off)
y_pred = lin_reg.predict(X_test_off)
y_pred[y_pred < 0] = 0
total_error = np.sum(np.abs(y_pred - y))
mean_error = np.mean(np.abs(y_pred - y))

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

############# 
# y_pred = y2014
# y2 = Data[:,-2]
# a = np.abs(y2-y)
# print np.mean(a)
# plotConfusionMatrix(y, y_pred)


# Random Forest:
# Number of Trees K: K = 30 -> 95%, K = 100 -> 96%, K = 
# K = 2000



# Fully Connected Net:
# After 100 epoqs achieved accuracy of 99.98% on training, Test_accuracy is 97% - Overfit
# y = y.T 
# model = fullyConnectedNet(X, y, epochs = 20)
# results = model.predict(X_test)
# y_pred = np.zeros([results.shape[0]])
# for i in xrange(results.shape[0]):
# 	y_pred[i] = np.argmax(results[i,:]).astype(int)


rest = np.ones(100)*np.mean(y_pred)


code.interact(local=dict(globals(), **locals()))

# T-SNE
# X_tSNE = tsne(X, 3, 50, 20.0);
# plot3d(X_tSNE,y,"tSNE-Plot")

### Evaluation
# difference = np.sum(np.absolute(y_pred - y)) 
# print difference

# Submission:
saveToCSV(Data, y_pred*y_zeros, "final2")

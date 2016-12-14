#Classification of S-Data
* features3.py extracts data and features from .csv file and converts them into .npy array for machine learning tasks
* run.py loads data as numpy array, splits data into train and test and performs algorithms
* plotLib.py plots Data rows as histograms. Originally, insights from these plots were planned to enable thresholding and convert continuous data into discrete labels for deep learning.
* nets.py contains Deep Learning logic and libraries. Not needed.

### Implementing various ML/DL algorithms - Performance Assessment

#### Using the simplest submission: y2014 = y2015:
* error is 11000 per feature.
* not very promissing

#### Using simple linear Regression:
* using only Linear Regression and subtracting the mean_error from y_pred, average error is 8000.

#### Using SVM to predict purchases and then apply simple linear Regression:
* works well on small dataset, slightly improving prediction
* SVM too computationally expensive for large dataset

#### Using random Forest Regression to predict purchases and then apply simple linear Regression:
* best performing approach so far
* winning submission of the hackathon


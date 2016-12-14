# implement a neural network for classification
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, ZeroPadding2D, MaxPooling2D, Dropout, Flatten
import numpy as np
import tensorflow as tf
tf.python.control_flow_ops = tf

def fullyConnectedNet(X, y, epochs):
	neurons = 100
	nb_features = X.shape[1]
	nb_classes
	model = Sequential([
		Dense(neurons, input_dim=nb_features, init='uniform'),
		Activation('relu'),
		Dense(neurons, init='uniform'),
		Activation('relu'),
		Dense(nb_classes, init='uniform'),
		Activation('softmax'),
	])

	# Compile model
	model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
	# mse does not work
	# Fit the model
	model.fit(X, y, nb_epoch=epochs, batch_size=100)

	# evaluate the model
	scores = model.evaluate(X, y)
	print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
	return model

def betterFullyConnectedNet(X, y, epochs):
	neurons = 100
	nb_features = X.shape[1]
	model = Sequential([
		Dense(neurons, input_dim=nb_features, init='uniform'),
		Activation('relu'),
		Dropout(.5),
		Dense(neurons, init='uniform'),
		Activation('relu'),
		Dropout(.25),
		Dense(8, init='uniform'),
		Activation('relu'),
		Dropout(.25),
		Dense(8, init='uniform'),
		Activation('relu'),
		Dropout(.25),
		Dense(8, init='uniform'),
		Activation('softmax'),
	])

	# Compile model
	model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
	# mse does not work
	# Fit the model
	model.fit(X, y, nb_epoch=epochs, batch_size=100)

	# evaluate the model
	scores = model.evaluate(X, y)
	print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
	return model
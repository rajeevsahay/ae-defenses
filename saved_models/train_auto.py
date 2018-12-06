import keras
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense
from keras.datasets import mnist
from keras.utils import np_utils
from keras import backend
import os
import numpy as np

#============================================================================================================
def get_data():
	#Import dataset and normalize to [0,1]
	(data_train, labels_train), (data_test, labels_test) = mnist.load_data()
	data_train = data_train/255.0
	data_test = data_test/255.0

	#Flatten dataset (New shape for training and testing set is (60000,784) and (10000, 784))
	data_train = data_train.reshape((len(data_train), np.prod(data_train.shape[1:])))
	data_test = data_test.reshape((len(data_test), np.prod(data_test.shape[1:])))

	#Create labels as one-hot vectors
	labels_train = keras.utils.np_utils.to_categorical(labels_train, num_classes=10)
	labels_test = keras.utils.np_utils.to_categorical(labels_test, num_classes=10)
	return [data_train, labels_train, data_test, labels_test]
#============================================================================================================
def train_autoencoder(data, path_to_save, num_layers = 80):
	[X_train, Y_train, X_test, Y_test] = data
	autoencoder = Sequential()
	autoencoder.add(Dense(num_layers, input_dim = 784, activation='relu', kernel_initializer = 'uniform'))
	autoencoder.add(Dense(784, activation='sigmoid', kernel_initializer = 'uniform'))

	autoencoder.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
	autoencoder.fit(X_train, X_train, epochs=100, batch_size=500) #train
	autoencoder.save(path_to_save)
	return autoencoder
#============================================================================================================
if __name__ == '__main__':
	data = get_data()
	num_layers = 80
	auto = train_autoencoder(data, 'autoencoders/{}_test.h5'.format(num_layers), num_layers=num_layers)

#Creating a fully-connected classifier using clean,
#uncorrupted MNIST data on ten digits

#Loss minimizes to 0.0081 over 20 epochs using cross entropy and adam
#Acieves accuracy of 97.88% on testing data

import numpy as np
import tensorflow as tf
from keras.models import Sequential
import keras
from keras.datasets import mnist
from keras.layers import Dense
from cleverhans.attacks import FastGradientMethod
from cleverhans.utils_keras import KerasModelWrapper
import time
from sp_func_svm import sp_project, sp_frontend
from keras.models import load_model

start_time = time.time()


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


#Create the model
def fc_model():

    model = Sequential()
    model.add(Dense(200, activation="relu", use_bias=True, kernel_initializer="normal", input_dim=784))
    model.add(Dense(200, activation="relu", kernel_initializer="normal"))
    model.add(Dense(100, activation="relu", kernel_initializer="normal"))
    model.add(Dense(10, activation="softmax", kernel_initializer="normal"))
    return model



model = fc_model()

#Model parameters
batch_size = 200

#Compile model using croo entropy as loss and adam as optimizer
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#Train model using input of clean and corrupted data and fit to clean reconstructions only
model.fit(data_train, labels_train, validation_data=(data_test, labels_test), epochs=20, batch_size=batch_size, shuffle=True)

#Save the model
model.save('fc-200-200-100-10.h5')

#Evaluate
scores = model.evaluate(data_test, labels_test)

#Print accuracy
print ("Accuracy: %.2f%%" %(scores[1]*100))

#Print runtime
print("Runtime is %s seconds" % (time.time() - start_time))

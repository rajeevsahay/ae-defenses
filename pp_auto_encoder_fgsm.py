import numpy as np
from keras.models import Sequential
from keras.datasets import mnist
from keras.layers import Dense
from keras.models import load_model
from keras import backend
from cleverhans.attacks import FastGradientMethod
from cleverhans.utils_keras import KerasModelWrapper
import matplotlib.pyplot as plt

#Loss minimizes to 0.0049 over 150 epochs using mean squared error and adam


#Load MNIST data and normalize to [0,1]
(data_train, _), (data_test, _) = mnist.load_data()
data_train = data_train/255.0
data_test = data_test/255.0

#Flatten dataset (New shape for training and testing set is (60000,784) and (10000, 784))
data_train = data_train.reshape((len(data_train), np.prod(data_train.shape[1:])))
data_test = data_test.reshape((len(data_test), np.prod(data_test.shape[1:])))

#Load classifier model whose gradients will be used to create adversarial examples
keras_model = load_model('fc-100-100-10.h5')
backend.set_learning_phase(False)

#Create adversarial examples on testing data
sess =  backend.get_session()
epsilon = 0.25
wrap = KerasModelWrapper(keras_model)
fgsm = FastGradientMethod(wrap, sess=sess)
adv_train_x = fgsm.generate_np(data_train, eps=epsilon, clip_min=0., clip_max=1.)
adv_test_x = fgsm.generate_np(data_test, eps=epsilon, clip_min=0., clip_max=1.)

#Total datasets
data_total_train = np.vstack([data_train, adv_train_x])
data_total_test = np.vstack([data_test, adv_test_x])

#Create labels that correspond to clean reconstructions
labels_total_train = np.vstack([data_train, data_train])
labels_total_test = np.vstack([data_test, data_test])

#Create the model
def autoencoder():

    model = Sequential()
    model.add(Dense(256, activation=None, use_bias=True, kernel_initializer="uniform", input_dim=784))
    model.add(Dense(128, activation=None, kernel_initializer="uniform"))
    model.add(Dense(64, activation=None, kernel_initializer="uniform"))
    model.add(Dense(128, activation=None, kernel_initializer="uniform"))
    model.add(Dense(256, activation=None, kernel_initializer="uniform"))
    model.add(Dense(784, activation="sigmoid", kernel_initializer="uniform"))
    return model



model = autoencoder()

#Compile model using mean squared error as loss and adam as optimizer
model.compile(loss='mean_squared_error', optimizer='adam')

#Train model using input of clean and corrupted data and fit to clean reconstructions only
model.fit(data_total_train, labels_total_train, validation_data=(data_total_test, labels_total_test), epochs=150, batch_size=256, shuffle=True)

#Save the model
model.save('pp_auto_encoder.h5')

#Predict reconstructions of test data
decoded_images = model.predict(data_total_test)

#Plot samples of first 15 perturbed images before and after reconstruction
n = 15  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(data_total_test[i+10000].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_images[i+10000].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()




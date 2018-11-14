import numpy as np
import keras
from keras import backend
from keras.datasets import mnist
from keras.models import load_model
from cleverhans.attacks import FastGradientMethod
from cleverhans.utils_keras import KerasModelWrapper


#Load training and testing data and normalize in [0, 1]
(data_train, labels_train), (data_test, labels_test) = mnist.load_data()
data_train = data_train/255.0
data_test = data_test/255.0

#Flatten dataset (New shape for training and testing set is (60000,784) and (10000, 784))
data_train = data_train.reshape((len(data_train), np.prod(data_train.shape[1:])))
data_test = data_test.reshape((len(data_test), np.prod(data_test.shape[1:])))

#Create labels as one-hot vectors
labels_train = keras.utils.np_utils.to_categorical(labels_train, num_classes=10)
labels_test = keras.utils.np_utils.to_categorical(labels_test, num_classes=10)

#Import trained classifer
backend.set_learning_phase(False)
fc_classifier = load_model('fc-100-100-10.h5')

#Evaluate on clean data
scores = fc_classifier.evaluate(data_test, labels_test)

#Print accuracy of unattacked, no defense testing set
print("Accuracy of clean data without any defense")
print ("Accuracy: %.2f%%" %(scores[1]*100))

#Create adversarial examples on testing data
sess =  backend.get_session()
epsilon = 0.25
wrap = KerasModelWrapper(fc_classifier)
fgsm = FastGradientMethod(wrap, sess=sess)
#adv_train_x = fgsm.generate_np(data_train, eps=epsilon, clip_min=0., clip_max=1.)
adv_test_x = fgsm.generate_np(data_test, eps=epsilon, clip_min=0., clip_max=1.)

#Evaluate model after attacking data
adv_acc = fc_classifier.evaluate(adv_test_x, labels_test)

#Print accuracy of attacked, no defense testing set
print("Accuracy of perturbed data without defense")
print ("Accuracy: %.2f%%" %(adv_acc[1]*100))

#Load pre-processing autoencoder
pp_ae = load_model('pp_auto_encoder_fgsm.h5')

#Run testing data through pre-processor
decoded_data = pp_ae.predict(adv_test_x)

#Evaluate accuracy of classifier after pre-processing
adv_scores = fc_classifier.evaluate(decoded_data, labels_test)

#Print accuracy of attacked data after preprocessing
print("Accuracy of perturbed data using DAE as Defense")
print ("Accuracy: %.2f%%" %(adv_scores[1]*100))




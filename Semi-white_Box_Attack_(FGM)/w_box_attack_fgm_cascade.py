import numpy as np
import keras
from keras.datasets import mnist
from keras.models import load_model
from keras import backend
from keras import backend as K
from cleverhans.utils_keras import KerasModelWrapper
from cleverhans.attacks import FastGradientMethod

#Load training and testing data and normalize to [0, 1]
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
#fc_classifier = load_model('fc-100-100-10.h5')

#Evaluate on clean data
scores = fc_classifier.evaluate(data_test, labels_test)

#Print accuracy of unattacked, no defense testing set
print("Accuracy of clean data without any defense")
print ("Accuracy: %.2f%%" %(scores[1]*100))

#Create adversarial examples according to l-2 norm on testing data
sess =  backend.get_session()
epsilon = 2.5
ord = 2
wrap = KerasModelWrapper(fc_classifier)
fgm = FastGradientMethod(wrap, sess=sess)
adv_train_x = fgm.generate_np(data_train, eps=epsilon, ord=ord, clip_min=0., clip_max=1.)
adv_test_x = fgm.generate_np(data_test, eps=epsilon, ord=ord, clip_min=0., clip_max=1.)

np.savetxt("adv_test_x.csv", adv_test_x, delimiter=",")

#Evaluate model after attacking data without pre-processing
adv_acc = fc_classifier.evaluate(adv_test_x, labels_test)

#Print accuracy of attacked, no defense testing set
print("Accuracy of perturbed data without defense")
print("Accuracy: %.2f%%" %(adv_acc[1]*100))

#Load pre-processing autoencoder
#pp_ae = load_model('pp_auto_encoder_fgm_l2_eps1.h5')
pp_ae = load_model('../Semi-white_Box_Attack_(FGSM)/pp_auto_encoder_fgsm.h5')

#Run testing data through pre-processor
decoded_data = pp_ae.predict(adv_test_x)

#Evaluate accuracy of classifier after pre-processing
adv_scores = fc_classifier.evaluate(decoded_data, labels_test)

#Print accuracy of attacked data after preprocessing
print("Accuracy of perturbed data using DAE as Defense")
print ("Accuracy: %.2f%%" %(adv_scores[1]*100))

#Load dimensionality reduction AE
red_dim_ae = load_model('../saved_models/autoencoders/784-60-784_100epochs.h5')

#Obtain reduced dimensions of perturbed testing set
ae_hidden_representation = K.function([red_dim_ae.layers[0].input], [red_dim_ae.layers[0].output])
hidden_representation = ae_hidden_representation([adv_test_x])

#Load classifier with input dimension of 60
red_dim_classifier = load_model('../saved_models/classifiers/60-100_100epochs.h5')

#Evaluate accuracy with reduced dimensions
red_dim_acc = red_dim_classifier.evaluate(hidden_representation, labels_test)

#Print accuracy of attacked data after reducing dimensions
print("Accuracy of perturbed data using dim. red. as Defense")
print ("Accuracy: %.2f%%" %(red_dim_acc[1]*100))

#Cascade defense
series_defense_data = ae_hidden_representation([decoded_data])
sereis_acc = red_dim_classifier.evaluate(series_defense_data, labels_test)

#Print accuracy of attacked data after series defense
print("Accuracy of perturbed data using series Defense")
print ("Accuracy: %.2f%%" %(sereis_acc[1]*100))

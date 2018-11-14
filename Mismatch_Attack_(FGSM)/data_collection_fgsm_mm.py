import numpy as np
import keras
from keras.datasets import mnist
from keras.models import load_model
from keras import backend
from cleverhans.attacks import FastGradientMethod
from cleverhans.utils_keras import KerasModelWrapper

e = np.linspace(0.01,0.30,30)
data = np.zeros([30, 4])

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
fc_classifier = load_model('fc-100-100-10.h5')

#Evaluate on clean data
scores = fc_classifier.evaluate(data_test, labels_test)

#Print accuracy of unattacked, no defense testing set
#print ("Accuracy: %.2f%%" %(scores[1]*100))


for idx, epsilon in enumerate(e):

    data[idx, 0] = epsilon
    data[idx, 1] = scores[1]*100

    #Create adversarial examples on testing data using gradients from mismatched classifier
    atk_classifier = load_model('fc-200-200-100-10.h5')
    backend.set_learning_phase(False)
    sess =  backend.get_session()
    #epsilon = 0.3
    wrap = KerasModelWrapper(atk_classifier)
    fgsm = FastGradientMethod(wrap, sess=sess)
    #adv_train_x = fgsm.generate_np(data_train, eps=epsilon, clip_min=0., clip_max=1.)
    adv_test_x = fgsm.generate_np(data_test, eps=epsilon, clip_min=0., clip_max=1.)


    #Evaluate model after attacking data
    adv_acc = fc_classifier.evaluate(adv_test_x, labels_test)

    #Print accuracy of attacked, no defense testing set
    #print ("Accuracy: %.2f%%" %(adv_acc[1]*100))
    data[idx, 2] = adv_acc[1]*100

    #Load pre-processing autoencoder trained using adversarial examples from original classifier
    pp_ae = load_model('pp_auto_encoder_mismatch.h5')

    #Run testing data through pre-processor
    decoded_data = pp_ae.predict(adv_test_x)

    #Evaluate accuracy of classifier after pre-processing
    adv_scores = fc_classifier.evaluate(decoded_data, labels_test)

    #Print accuracy of attacked data after preprocessing
    #print ("Accuracy: %.2f%%" %(adv_scores[1]*100))
    data[idx, 3] = adv_scores[1]*100

np.savetxt("fgsm_mismatch.csv", data, delimiter=",")

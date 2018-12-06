import numpy as np
import keras
from keras import backend
from keras.datasets import mnist
from keras.models import load_model
from keras import backend as K
from cleverhans.attacks import FastGradientMethod
from cleverhans.utils_keras import KerasModelWrapper


e = np.linspace(0.01,0.50,50)
data = np.zeros([50, 16])


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

#Setup adversairal attack
sess =  backend.get_session()
wrap = KerasModelWrapper(fc_classifier)
fgsm = FastGradientMethod(wrap, sess=sess)

#Load pre-processing autoencoders
pp_ae_eps25 = load_model('pp_auto_encoder_fgsm.h5')

#Load dimensionality reduction AE
red_dim_ae_20 = load_model('../saved_models/autoencoders/784-20-784_100epochs.h5')
red_dim_ae_40 = load_model('../saved_models/autoencoders/784-40-784_100epochs.h5')
red_dim_ae_60 = load_model('../saved_models/autoencoders/784-60-784_100epochs.h5')
red_dim_ae_80 = load_model('../saved_models/autoencoders/784-80-784_100epochs.h5')
red_dim_ae_100 = load_model('../saved_models/autoencoders/784-100-784_100epochs.h5')
red_dim_ae_331 = load_model('../saved_models/autoencoders/784-331-784_100epochs.h5')

#Indicate how to obtain hidden representation of AE
ae_hidden_representation_20 = K.function([red_dim_ae_20.layers[0].input], [red_dim_ae_20.layers[0].output])
ae_hidden_representation_40 = K.function([red_dim_ae_40.layers[0].input], [red_dim_ae_40.layers[0].output])
ae_hidden_representation_60 = K.function([red_dim_ae_60.layers[0].input], [red_dim_ae_60.layers[0].output])
ae_hidden_representation_80 = K.function([red_dim_ae_80.layers[0].input], [red_dim_ae_80.layers[0].output])
ae_hidden_representation_100 = K.function([red_dim_ae_100.layers[0].input], [red_dim_ae_100.layers[0].output])
ae_hidden_representation_331 = K.function([red_dim_ae_331.layers[0].input], [red_dim_ae_331.layers[0].output])

#Load classifier with input dimension corresponding to AE
red_dim_classifier_20 = load_model('../saved_models/classifiers/20-100_100epochs.h5')
red_dim_classifier_40 = load_model('../saved_models/classifiers/40-100_100epochs.h5')
red_dim_classifier_60 = load_model('../saved_models/classifiers/60-100_100epochs.h5')
red_dim_classifier_80 = load_model('../saved_models/classifiers/80-100_100epochs.h5')
red_dim_classifier_100 = load_model('../saved_models/classifiers/100-100_100epochs.h5')
red_dim_classifier_331 = load_model('../saved_models/classifiers/331-100_100epochs.h5')


for idx, epsilon in enumerate(e):
    print(idx)
    data[idx, 0] = epsilon
    data[idx, 1] = scores[1]*100 #No attack, no defense

    #Create adversarial examples on testing data
    adv_test_x = fgsm.generate_np(data_test, eps=epsilon, clip_min=0., clip_max=1.)

    #Evaluate model after attacking data
    adv_acc = fc_classifier.evaluate(adv_test_x, labels_test)
    data[idx, 2] = adv_acc[1]*100 #attack, no defense

    #Run testing data through DAE pre-processor trained on eps=0.25
    decoded_data_eps25 = pp_ae_eps25.predict(adv_test_x)

    #Evaluate accuracy of classifier after pre-processing
    adv_scores_eps25 = fc_classifier.evaluate(decoded_data_eps25, labels_test)
    data[idx,3] = adv_scores_eps25[1]*100 #attacked and defended using DAE trained on eps=0.25

    #Obtain reduced dimensions of perturbed testing set
    hidden_representation_20 = ae_hidden_representation_20([adv_test_x])
    hidden_representation_40 = ae_hidden_representation_40([adv_test_x])
    hidden_representation_60 = ae_hidden_representation_60([adv_test_x])
    hidden_representation_80 = ae_hidden_representation_80([adv_test_x])
    hidden_representation_100 = ae_hidden_representation_100([adv_test_x])
    hidden_representation_331 = ae_hidden_representation_331([adv_test_x])

    #Evaluate accuracy with reduced dimensions
    red_dim_acc_20 = red_dim_classifier_20.evaluate(hidden_representation_20, labels_test)
    data[idx,4] = red_dim_acc_20[1]*100 #attacked and defended using reduced dimension of 20
    red_dim_acc_40 = red_dim_classifier_40.evaluate(hidden_representation_40, labels_test)
    data[idx,5] = red_dim_acc_40[1]*100 #attacked and defended using reduced dimension of 40
    red_dim_acc_60 = red_dim_classifier_60.evaluate(hidden_representation_60, labels_test)
    data[idx,6] = red_dim_acc_60[1]*100 #attacked and defended using reduced dimension of 60
    red_dim_acc_80 = red_dim_classifier_80.evaluate(hidden_representation_80, labels_test)
    data[idx,7] = red_dim_acc_80[1]*100 #attacked and defended using reduced dimension of 80
    red_dim_acc_100 = red_dim_classifier_100.evaluate(hidden_representation_100, labels_test)
    data[idx,8] = red_dim_acc_100[1]*100 #attacked and defended using reduced dimension of 100
    red_dim_acc_331 = red_dim_classifier_331.evaluate(hidden_representation_331, labels_test)
    data[idx,9] = red_dim_acc_331[1]*100 #attacked and defended using reduced dimension of 331


    #Cascade defenses
    series_defense_data_20 = ae_hidden_representation_20([decoded_data_eps25])
    sereis_acc_20 = red_dim_classifier_20.evaluate(series_defense_data_20, labels_test)
    data[idx,10] = sereis_acc_20[1]*100 #attacked and defended using series defense with DAE trained on eps=0.25 and K=20
    series_defense_data_40 = ae_hidden_representation_40([decoded_data_eps25])
    sereis_acc_40 = red_dim_classifier_40.evaluate(series_defense_data_40, labels_test)
    data[idx,11] = sereis_acc_40[1]*100 #attacked and defended using series defense with DAE trained on eps=0.25 and K=40
    series_defense_data_60 = ae_hidden_representation_60([decoded_data_eps25])
    sereis_acc_60 = red_dim_classifier_60.evaluate(series_defense_data_60, labels_test)
    data[idx,12] = sereis_acc_60[1]*100 #attacked and defended using series defense with DAE trained on eps=0.25 and K=60
    series_defense_data_80 = ae_hidden_representation_80([decoded_data_eps25])
    sereis_acc_80 = red_dim_classifier_80.evaluate(series_defense_data_80, labels_test)
    data[idx,13] = sereis_acc_80[1]*100 #attacked and defended using series defense with DAE trained on eps=0.25 and K=80
    series_defense_data_100 = ae_hidden_representation_100([decoded_data_eps25])
    sereis_acc_100 = red_dim_classifier_100.evaluate(series_defense_data_100, labels_test)
    data[idx,14] = sereis_acc_100[1]*100 #attacked and defended using series defense with DAE trained on eps=0.25 and K=100
    series_defense_data_331 = ae_hidden_representation_331([decoded_data_eps25])
    sereis_acc_331 = red_dim_classifier_331.evaluate(series_defense_data_331, labels_test)
    data[idx,15] = sereis_acc_331[1]*100 #attacked and defended using series defense with DAE trained on eps=0.25 and K=331


np.savetxt("fgsm_whitebox_data", data, delimiter=",")

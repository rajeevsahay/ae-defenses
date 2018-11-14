import keras.backend as K
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.datasets import mnist
from keras.utils import np_utils
from keras import backend
from cleverhans.attacks import FastGradientMethod
from cleverhans.utils_keras import KerasModelWrapper
from sklearn.decomposition import PCA
import os
import numpy as np
import matplotlib.pyplot as plt

path = ''
#STANDARDIZATION; 100 epochs
#API: data = [X_train, Y_train, X_test, Y_test]
#kernel_initializer
#============================================================================================================
def get_data():
	'''(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
	X_train = X_train/255.0
	X_test = X_test/255.0
	X_train = X_train.reshape((len(X_train), np.prod(X_train.shape[1:])))
	X_test = X_test.reshape((len(X_test), np.prod(X_test.shape[1:])))
	#X_train = X_train.astype('float32')
	#X_test = X_test.astype('float32')
	#Preprocess class labels
	Y_train = np_utils.to_categorical(Y_train, num_classes=10)
	Y_test = np_utils.to_categorical(Y_test, num_classes=10)
	#X_train = X_train[:-10000]; Y_train = Y_train[:-10000]
	return [X_train, Y_train, X_test, Y_test]'''
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
def gen_model(data, epochs, num_layers, fileName = None):
	'''generates and saves classifier trained on uncorrupted data'''
	[X_train, Y_train, X_test, Y_test] = data
	print('in gen_model, num_layers = {}, X_train.shape = {}'.format(num_layers, np.array(X_train).shape))
	if fileName is None:
		fileName = path + 'saved_models/FGS/{}-100_{}epochs.h5'.format(num_layers, epochs)

	model = Sequential()
	model.add(Dense(100, activation="relu", use_bias=True, kernel_initializer="normal", input_dim=num_layers))
	model.add(Dense(100, activation="relu", kernel_initializer="normal"))
	model.add(Dense(10, activation="softmax", kernel_initializer="normal"))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=100, batch_size=200, shuffle=True)
	model.save(fileName)
	return model
	'''classifier = Sequential()
	classifier.add(Dense(100, input_dim=num_layers, activation='relu', kernel_initializer = 'normal'))
	classifier.add(Dense(100, activation='relu', kernel_initializer = 'normal'))
	classifier.add(Dense(10, activation='softmax', kernel_initializer = 'normal'))
	
	optimizer = keras.optimizers.SGD(lr = 0.01, momentum = 0.9)
	classifier.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	classifier.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=epochs, batch_size=200) #training'''

	'''classifier = Sequential()
	classifier.add(Dense(100, input_dim=784, activation='sigmoid', kernel_initializer = 'normal', use_bias = True))
	classifier.add(Dense(100, activation='sigmoid', kernel_initializer = 'normal'))
	classifier.add(Dense(10, activation='softmax', kernel_initializer = 'normal'))
	
	optimizer = keras.optimizers.SGD(lr = 0.01, momentum = 0.9)
	classifier.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
	classifier.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=epochs, batch_size=500, shuffle=True) #training
	classifier.save(fileName)
	return classifier'''
#============================================================================================================
def perturb(data, epochs = 10, epsilon = 0.25): #nn architecture, attack
	'''trains its own model on train data, perturbs test data'''
	[X_train, Y_train, X_test, Y_test] = data
	#fileName = path + 'saved_models/adv_models/FGS/{}-100_{}epochs.h5'.format(784, epochs)
	fileName = path + 'saved_models/classifiers/{}-100_{}epochs.h5'.format(784, epochs) #classifier fileName
	#fileName = path + 'ML-Science/fc-100-100-10.h5'
	#fileName = path + 'bla.h5'
	#fileName = path + 'saved_models/classifiers/{}-100_{}epochs.h5'.format(784, 100) #classifier fileName
	#fileName = 'hehe.h5'
		
	if not os.path.exists(fileName):
		gen_model(data, epochs, 784, fileName)
	model = keras.models.load_model(fileName)
	#backend.set_learning_phase(False)
	sess =  backend.get_session()
	wrap = KerasModelWrapper(model)
	fgsm = FastGradientMethod(wrap, sess=sess)
	adv_x = fgsm.generate_np(X_test, eps=epsilon, ord = 2, clip_min=0., clip_max=1.) #change ord to np.inf or 2, depending on fgs or fg
	return adv_x
	#if visualize == True:
	#print(adv_x.shape) #(60000,784)
#============================================================================================================
def perturb_arch_mismatch(data, epochs = 10, epsilon = 0.25): #200-200-100-10
	'''trains its own model on train data, perturbs test data'''
	[X_train, Y_train, X_test, Y_test] = data
	fileName = path + 'saved_models/adv_models/arch_mismatch/{}-30-10_{}epochs.h5'.format(784, epochs)
	if not os.path.exists(fileName):
		'''model = Sequential()
		model.add(Dense(200, input_dim=num_layers, activation='sigmoid'))
		model.add(Dense(200, activation='sigmoid'))
		model.add(Dense(100, activation='sigmoid'))
		model.add(Dense(10, activation='softmax'))
		
		optimizer = keras.optimizers.SGD(lr = 0.01, momentum = 0.9)
		model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
		model.fit(X_train, Y_train, epochs=epochs, batch_size=500) #training
		model.save(fileName)'''
		model = Sequential()
		model.add(Dense(200, activation="relu", use_bias=True, kernel_initializer="normal", input_dim=784))
		model.add(Dense(200, activation="relu", kernel_initializer="normal"))
		model.add(Dense(100, activation="relu", kernel_initializer="normal"))
		model.add(Dense(10, activation="softmax", kernel_initializer="normal"))
		model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
		model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=100, batch_size=200, shuffle=True)
		model.save(fileName)
			
	else:
		model = keras.models.load_model(fileName)
	#backend.set_learning_phase(False)
	sess =  backend.get_session()
	wrap = KerasModelWrapper(model)
	fgsm = FastGradientMethod(wrap, sess=sess)
	adv_x = fgsm.generate_np(X_test, eps=epsilon, ord=2,  clip_min=0., clip_max=1.)
	return adv_x
	#if visualize == True:
	#print(adv_x.shape) #(60000,784)
#============================================================================================================
def train_autoencoder(data, path_to_save, epochs = 100, num_layers = 392):
	'''path_to_save: path of .h5 file to store trained autoencoder in'''
	[X_train, Y_train, X_test, Y_test] = data
	autoencoder = Sequential()
	autoencoder.add(Dense(num_layers, input_dim=784, activation='relu'))
	#autoencoder.add(Dense(196, activation='relu'))
	#autoencoder.add(Dense(392, activation='relu'))
	autoencoder.add(Dense(784, activation='sigmoid'))
	
	autoencoder.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
	autoencoder.fit(X_train, X_train, epochs=epochs, batch_size=500) #train
	autoencoder.save(path_to_save)
	return autoencoder

	# check if it predicts correctly
	#predictions = autoencoder.predict(X_test[:10])
	#X_test *= 255
	#predictions *= 255
	#Image.fromarray(X_test[0].reshape(28,28)).show()
	#Image.fromarray(predictions[0].reshape(28,28)).show()
#============================================================================================================
def train_test_normal_classifier(data, path_to_save, epochs = 100, num_layers = 392):
	'''trains and tests an FC100-100-10 with an input layer of size 784'''
	[X_train, Y_train, X_test, Y_test] = data
	#FC 100-100-10
	classifier = Sequential()
	classifier.add(Dense(100, input_dim=784, activation='sigmoid'))
	classifier.add(Dense(100, activation='sigmoid'))
	classifier.add(Dense(10, activation='softmax'))
	
	optimizer = keras.optimizers.SGD(lr = 0.01, momentum = 0.9)
	classifier.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
	classifier.fit(X_train, Y_train, epochs=epochs, batch_size=500) #training
	#classifier.save(path_to_save)
	
	predictions = classifier.predict(perturb(data))
	#predictions = classifier.predict(X_test)
	for i in range(len(predictions)):
		item = predictions[i]
		predictions[i] = list(map(lambda x: 0 if x <= 0.5 else 1, item))
	score = 0
	for i in range(len(predictions)):
		if np.array_equal(predictions[i], Y_test[i]):
			score += 1
	print('Score with NOT reduced dimension = {}, accuracy = {}'.format(score, np.float(score)*100.0/len(X_test)))
#============================================================================================================
def train_test_pca_classifier(data, path_to_save, epochs = 100, num_layers = 392, epsilon = 0.25): #nn architecture, attack
	'''first perturb, then do pca. otherwise accuracy is 0 if first pca, then perturb'''
	[X_train, Y_train, X_test, Y_test] = data
	fileName = path + 'saved_models/FGS/{}-100_{}epochs.h5'.format(num_layers, epochs)

	pca = PCA(n_components = num_layers)
	pca.fit(X_train.T)
	reduced_Xtrain = pca.components_.T 
	pca.fit(X_test.T)
	reduced_Xtest = pca.components_.T 
	data_to_send = [reduced_Xtrain, Y_train, reduced_Xtest, Y_test]
	#if not os.path.exists(fileName):
	#	gen_model(data_to_send, epochs, num_layers, fileName = fileName)
	classifier = Sequential()
	classifier.add(Dense(100, input_dim=num_layers, activation='sigmoid'))
	classifier.add(Dense(100, activation='sigmoid'))
	classifier.add(Dense(10, activation='softmax'))
	
	optimizer = keras.optimizers.SGD(lr = 0.01, momentum = 0.9)
	classifier.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
	classifier.fit(reduced_Xtrain, Y_train, epochs=epochs, batch_size=500) #training
	#mod_Xtrain = np.array(list(map(lambda x: x[:784], X_train)))
	#classifier.fit(mod_Xtrain, Y_train, epochs=epochs, batch_size=500) #training

	#classifier.save(fileName)

	#classifier = keras.models.load_model(fileName)
	#pca.fit(perturb(data_to_send, epochs, num_layers, epsilon = epsilon).T)  #nn architecture, attack
	#pca.fit(perturb(data, epochs, epsilon = epsilon).T)  #nn architecture, attack
	#pca2 = PCA(n_components = num_layers)
	#pca2.fit(X_test.T)
	#predictions = classifier.predict(pca.components_.T) #test

	#perturbed = perturb(data, epochs, epsilon = epsilon)
	#mod_perturbed = np.array(list(map(lambda x: x[:784], perturbed)))
	#mod_Xtest = np.array(list(map(lambda x: x[:784], X_test)))
	predictions = classifier.predict(reduced_Xtest) #test

	#print(predictions[:5])

	for i in range(len(predictions)):
		item = predictions[i]
		predictions[i] = list(map(lambda x: 0 if x <= 0.5 else 1, item))
	#print('predictions = ', predictions[:4])
	#print('Y_test = ', Y_test[:4])

	#print(Y_test[:5])
	score = 0
	for i in range(len(predictions)):
		if np.array_equal(predictions[i], Y_test[i]):
			score += 1
	to_write = '{} NN architecture trained with {} epochs with PCA-reduced dimension,\
attacked with {} attack with epsilon = {}. Score = {}, accuracy = {}\n'\
		  .format(str(num_layers) + '-100-100-10', epochs, 'FGS', epsilon, score, np.float(score)/len(X_test)*100.0)
	print(to_write)
	if epsilon == 0.25:
		to_write += '\n'
	with open(path + 'info.txt', 'a') as fh:
		fh.write(to_write)
#============================================================================================================
def train_test_red_classifier(data, epochs = 10, num_layers = 392, epsilon = 0.25, autoencoder_epochs = 15):
	'''trains and tests classifier with input layer of reduced dimension'''
	[X_train, Y_train, X_test, Y_test] = data
	fileName = path + 'saved_models/classifiers/{}-100_{}epochs.h5'.format(num_layers, epochs) #classifier fileName
	#fileName = 'hehe.h5'
	#fileName = 'bla.h5'
	if num_layers != 784:
		autoencoderFn = path + 'saved_models/autoencoders/784-{}-784_{}epochs.h5'.format(num_layers, autoencoder_epochs)
		if not os.path.exists(autoencoderFn):
			autoencoder = train_autoencoder(data, autoencoderFn, epochs = autoencoder_epochs, num_layers = num_layers)
		else:
			backend.set_learning_phase(False)
			autoencoder = keras.models.load_model(autoencoderFn)
		get_hidden_layer_output = K.function([autoencoder.layers[0].input], [autoencoder.layers[0].output])
		red_dim = get_hidden_layer_output([X_train]) #input into classifier for training
		red_dim_test = get_hidden_layer_output([X_test]) #input into classifier for training
	else: #no defense
		red_dim = X_train; red_dim_test = X_test
	#-----------------------------------------------------------------------------------------------------------
	#FC 100-100-10
	if not os.path.exists(fileName):
		'''classifier = Sequential()
		classifier.add(Dense(100, input_dim=num_layers, activation='sigmoid'))
		classifier.add(Dense(100, activation='sigmoid'))
		classifier.add(Dense(10, activation='softmax'))

		optimizer = keras.optimizers.SGD(lr = 0.01, momentum = 0.9, nesterov=True)
		classifier.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
		classifier.fit(red_dim, Y_train, epochs=epochs, batch_size=500) #training
		classifier.save(fileName)'''
		classifier = gen_model([red_dim, Y_train, red_dim_test, Y_test], epochs, num_layers, fileName)
	else:
		backend.set_learning_phase(False)
		classifier = keras.models.load_model(fileName)
		
	if epsilon != 0:
		perturbed_x = perturb_arch_mismatch(data, epochs, epsilon) #change to perturb or perturb_arch_mismatch depending on white_box or black_box
	else: #no perturbation
		perturbed_x = X_test
	if num_layers != 784:
		in_classifier = get_hidden_layer_output([perturbed_x]) #input into classifier for testing
	else: #no defense
		in_classifier = perturbed_x
	print('NUM_LAYERS = {}, CLASSIFIER FILENAME = {}'.format(num_layers, fileName))
	accuracy = classifier.evaluate(in_classifier, Y_test)
	'''predictions = classifier.predict(in_classifier)

	for i in range(len(predictions)):
		item = predictions[i]
		predictions[i] = list(map(lambda x: 0 if x <= 0.5 else 1, item))

	score = 0
	for i in range(len(predictions)):
		if np.array_equal(predictions[i], Y_test[i]):
			score += 1
	accuracy = np.float(score)*100.0/len(X_test)'''
	#print('accuracy = ', accuracy[1]*100)
	return accuracy[1]*100
#============================================================================================================
def main():
	data = get_data()
	#perturb(data[0])
	#train_autoencoder(data, path + 'saved_models/red_dim.h5')
	#train_test_normal_classifier(data, path + 'saved_models/fc100.h5')
	train_test_red_classifier(data, epochs = 100, num_layers = 40, epsilon = 0, autoencoder_epochs = 15)
	#train_test_pca_classifier(data, path + 'saved_models/fc100_392in.h5')
	#train_test_pca_classifier(data, path + 'saved_models/dont_care.h5')

#============================================================================================================
def experiment(epochs = 10, func = train_test_red_classifier):
	num_layer_arr = [784, 331, 100, 80, 60, 40, 20]
	#num_layer_arr = [80, 60, 40, 20]
	#num_layer_arr = np.linspace(start=20, stop=800, num=40)
	#epsilon_arr = [0.05, 0.1, 0.15, 0.2, 0.25]
	epsilon_arr = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
	#epsilon_arr = [0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25]
	#epsilon_arr = np.linspace(start=0.01, stop=0.5, num=50)
	'''def plot(result, ylabel = 'accuracy'):
		for i in range(len(num_layer_arr)):
			plt.plot(epsilon_arr, result[i,:], label = str(num_layer_arr[i]))
		plt.xlabel('epsilon')
		plt.ylabel(ylabel + ' (%)')
		plt.title('White box FGS attack on FC100-100-10 NN on MNIST with autoencoder dimensionality reduction defense')
		plt.savefig(path + 'figures/{}.png'.format(ylabel))'''
	data = get_data()
	accuracy = []; adv_succ = []
	for num_layers in num_layer_arr:
		to_write = ''
		baseline_accuracy = train_test_red_classifier(data, epochs = epochs, num_layers = num_layers, epsilon = 0, autoencoder_epochs = 100) #no attack
		to_write += '{} NN architecture trained with {} epochs with autoencoder-reduced dimension,attacked with {} attack with epsilon = {}. Accuracy = {}, adversarial success = {}\n'.format(str(num_layers) + '-100-100-10', epochs, 'FG', 0, baseline_accuracy, 0)
		for epsilon in epsilon_arr:
			acc = train_test_red_classifier(data, epochs = epochs, num_layers = num_layers, epsilon = epsilon, autoencoder_epochs = 100)
			accuracy.append(acc)
			adv_succ.append((1.0 - (acc/baseline_accuracy))*100)
			to_write += '{} NN architecture trained with {} epochs with autoencoder-reduced dimension,attacked with {} attack with epsilon = {}. Accuracy = {}, adversarial success = {}\n'.format(str(num_layers) + '-100-100-10', epochs, 'FG', epsilon, acc, (1.0 - (acc/baseline_accuracy))*100)
		print(to_write)
		#with open('new_info_{}auto_{}class_FGS.txt'.format(100, epochs), 'a') as fh:
		with open('info_{}.txt'.format('bb_fg'), 'a') as fh:
			fh.write(to_write + '\n')
		#np.savetxt(X=np.vstack(epsilon_arr, accuracy), fname='{}layers_data.csv'.format(num_layers), delimiter=',')
#============================================================================================================
if __name__ == '__main__':
	#main()
	experiment(epochs = 100)

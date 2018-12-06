# ML-Science
This repository has code to illustrate mitigation of adversarial attacks using autoencoders. The possibly perturbed data is denoised using a Denoising Autoencoder, after which it is compressed using another autoencoder before being input into the classifier.

There are four directories containing scripts and trained models to illustrate one of four scenarios: Mismatch Attack FGSM, Mismatch Attack FGM, Semi White Box Attack FGSM, and Semi White Box Attack FGM. The saved models are <explain. example: denoising autoencoders>

The saved_models directory contains saved autoencoders to reduce the dimension of data, and classifiers which are trained using these reduced dimensions. There is also a train_auto.py script to generate an autoencoder to reduce the data to a specified number of dimensions.

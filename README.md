# ML-Science
This repository has code to illustrate mitigation of adversarial attacks using autoencoders. The possibly perturbed data is denoised using a Denoising Autoencoder, after which it is compressed using another autoencoder before being input into the classifier. The work here supplements the conclusions from our paper, "Combatting Adversarial Attacks through Denoising and Dimensionality Reduction: A Cascaded Autoencoder Approach." 

There are four directories containing scripts and trained models to illustrate one of four scenarios: Mismatch Attack FGS attack, Mismatch Attack FG attack, Semi White Box Attack FGS attack, and Semi White Box Attack FG attack. The saved models are fc-100-100-10.h5 (one of these models is in every directory corresponding to the defender's classifier), fc-200-200-100-10.h5 (the attacker's model in the black box case), fc-200-200-10 (another attacker's model in the black box case), pp_auto_encoder_mismatch.h5 (denoising autoencoder defense for black box FSG and black box FG attack), and pp_auto_encoder_fgsm.h5 (denoising autoencoder defense for semi-white box FGS and semi-white box FG attack). 

The saved_models directory contains saved autoencoders to reduce the dimension of data, and classifiers which are trained using these reduced dimensions. There is also a train_auto.py script to generate an autoencoder to reduce the data to a specified number of dimensions.

Finally, to obtain the resutls from our experiments, run the following scripts: 
Semi-white_Box_Attack_(FGM)/w_box_attack_fgm_cascade.py
Semi-white_Box_Attack_(FGSM)/w_box_attack_fgsm_cascade.py
Mismatch_attack_(FGM)/bl_box_fgm.py
Mismatch_Attack_(FGSM)/bl_box_attack_fgsm.py

All the needed models already exist in the directories. However, to generate your own models, adjust the parameters accordingly in the .py files and ensure the 'model.save()' line is uncommented. 

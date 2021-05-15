### Objective
This repository contains all files needed to do implement a system that can evaluate basic handwritten mathematical expressions from images, involving digits and basic mathematcial operators like addition (+), subraction (-) and multiplication (x).

### machine leanring models used
To create the system, two sets of data and machine learning models were created
- Version 1 models were trained on 20,000 samples from the MNIST dataset and all the images belonging to mathematical operators from the HaSy dataset. (22k images) 
Version 2
- Version 2 models were trained on 60,000 samples from the MNIST dataset and all the images belonging to to mathematical operators from a Kaggle dataset. (83k images)

### Files
1. Equation Preprocessing.ipynb : Used to convert images of handwritten equations to a .npy file, for final testing of models
2. Final Testing.ipynb : For final testing, to see how each model (and each version) performs on few handwritten equations
3. MLP models.ipynb : Contains the code required to train a simple MLP model using keras
4. Random forest and adaboost.ipynb : Contains code required to train Random forest and multi stage adaboost model for character recognition
5. preprocess.py : Our module which contains various functions including prediction, preprocessing and so on. 

More details can be found in https://github.com/JoseJoy249/Handwritten-Expression-Evaluation/blob/master/evaluating-handwritten-math.pdf

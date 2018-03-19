The repository contains all files needed to do basic handwritten mathematical expression evaluation, involving digits 0-9 and addition (+), subraction (-) and multiplication (x).

Two versions of data and models
Version 1
● Version 1 models were trained on 20,000 samples from the MNIST dataset and all the images belonging to symbols +,- and times(x) from the HaSy dataset. (22k images) 
Version 2
● Version 2 models were trained on 60,000 samples from the MNIST dataset and all the images belonging to symbols +,- and times(x) from a Kaggle dataset. (83k images)

Codes
1. Equation Preprocessing.ipynb : Used to convert images of handwritten equations to a .npy file, for final testing of models
2. Final Testing.ipynb : For final testing, to see how each model (and each version) performs on few handwritten equations
3. MLP models.ipynb : Contains the code required to train a simple MLP model using keras
4. Random forest and adaboost.ipynb : Contains code required to train Random forest and multi stage adaboost model for character recognition
5. preprocess.py : Our module which contains various functions including prediction, preprocessing and so on. 

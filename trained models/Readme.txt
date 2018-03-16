
Models trained on data_ver1 (20000 MNIST + HASY [+,-,times] )
1. RFmodel_ver1.1
Trained using raw data_ver2 (training images not binary). 2000 estimators, class balanced. 19.4 % test error

2. Adaboost_stage1_ver1.1
Trained on raw data_ver2(training images not binary). 250 estimators to classify whether digit or symbol. Test error  0

3. Adaboost_digits_ver1.1
Trained on raw data_ver2(training images not binary). 500 estimators to classify digits. 

4. Adaboost_chars_ver1.1
Trained on raw data_ver2(training images not binary). 500 estimators to classify digits. 
stage 2 Test error 18.1%

5. MLP_singlestage_ver1.1
Trained on raw data_ver1.1, 2 hidden layers [258,128,64] ,  Test error 2%




Models trained on data_ver2 (60000 MNIST + Kaggle [+,-,times] )

1. RFmodel_ver2.1
Trained using raw data_ver2 (training images not binary). 2000 estimators, class balanced. 18.75 % test error

2. Adaboost_stage1_ver2.1
Trained on raw data_ver2(training images not binary). 250 estimators to classify whether digit or symbol. Test error  0

3. Adaboost_digits_ver2.1
Trained on raw data_ver2(training images not binary). 500 estimators to classify digits. Test error 

4. Adaboost_chars_ver2.1
Trained on raw data_ver2(training images not binary). 500 estimators to classify digits. Stage 2 Test error 17 %

5. MLP_singlestage_ver2.1
Trained on raw data_ver2.1, 3 hidden layers [256,128,64] ,  Test error 1.3%
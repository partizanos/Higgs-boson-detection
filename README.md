# Higgs boson identification: Machine Learning Approach

Library used: NumPy

In this repository one can find all the code written for the project as well as the paper which briefly explains our approach. 

We provide the following notebooks which describe our full approach:
* EDA & Feature importance.ipynb - The notebook summurizes the exploratory data analysis and feature importance analysis done before training any models.
* Modeling.ipynb - In the following notebook we present the full deployment of our model, from hyperparameter tuning to cross validation. At the end the accuracy results on the train set are displayed. 

The notebooks use several Python files, which contain the implementation of all algorithms and general functions used for the project.
* data_processing.py - Contains all functions that are used within our data processing part of the code.
* feature_importance.py - Contains methods such as Riemann approximation and Gaussian test, which serve a purpose of fining the important features
* implementations.py - This is the file, which contain the optimization algorithms of our project. Apart from the 6 required functions we have implemented several advanced method which help us achieve our best score
* objective_functions.py - All objective functions such as 'mse_loss' and 'logit_loss' plus the functions for calculating the gradient are located in this file
* proj1_helpers.py - Functions provided by the teaching team for execution of simple data loading and creation of submission
* extra_helpers.py - General function that we implement to simplify our data processing and modeling.
* run.py - Runing this file loads our best model and tests its accuracy. 
* run_functions.py - Functions to avoid code repetition in our run.py file.

Additionally, we provide the full colection of .csv files used for the project in the `data` folder. It contains the following information:
* train.csv - labeled Higgs boson data used for training the model.
* test.csv - dataset for testing the accuracy of the created model.
* `run` folder - contains all optimized hyper-parameters and characteristics of the final model. Used for run file, which instead of finding the optimal parameters, loads them directly, which speeds up the processes of training and testing the final model.
* case 0-3 - pictures of the evaluation of our 4 final models
* case_0-3_CV5_statistics_median_null_var_pairwise_bias_scaling_orth_pol_1_to_20_1e4.csv - final result of the cross validation for each model. Used in `Plots.ipynb` to create the plots quickly.

# Higgs boson identification: Machine Learning Approach

Python version: 3
Libraries: - NumPy
           - Matplotlib

In this repository one can find all the code written for the project as well as the paper which briefly explains our approach. 

We provide the following notebooks which describe our full approach:
* EDA & Feature importance.ipynb - The notebook summarizes the exploratory data analysis and feature importance analysis done before training any models.
* Modeling.ipynb - In the following notebook we present the full deployment of our model, from hyperparameter tuning to predictions.

The notebooks use several Python files, which contain the implementation of all algorithms and general functions used for the project.
* data_processing.py - Contains all functions that are used within our data processing part of the code.
* feature_importance.py - Contains methods such as Riemann approximation and Gaussian test, which serve a purpose of finding the important features.
* implementations.py - This is the file, which contain the optimization algorithms of our project. Apart from the 6 required functions we have implemented several advanced methods which help us achieve our best score.
* objective_functions.py - All objective functions such as 'mse_loss' and 'logit_loss' plus the functions for calculating the gradient are located in this file.
* proj1_helpers.py - Functions provided by the teaching team for execution of simple data loading and creation of submission.
* extra_helpers.py - General function that we implement to simplify our data processing and modeling.
* run.py - Running this file loads our best model and produce the predictions for the test set. 
* run_functions.py - Functions to avoid code repetition in our run.py file.

Additionally, we provide the full collection of .csv files used for the project in the `data` folder. It contains the following information:
* train.csv - labeled Higgs boson data used for training the models.
* test.csv - dataset for for which we have to predict labels.
* `run` folder - contains all optimized hyper-parameters and characteristics of the final models. Used for run file, which instead of finding the optimal parameters, loads them directly, which speeds up the processes.

import numpy as np
from proj1_helpers import *
from data_processing import *
from objective_functions import *

def import_parameters(w_path, threshold_path, null_variance_index_path, mean_path, std_path, eigenvectors_path):
    """Importing relevant parameters for predictions."""
    w = np.genfromtxt(w_path, delimiter=",", skip_header=0, dtype=np.complex128)
    thresh = np.genfromtxt(threshold_path, delimiter=",", skip_header=0)[0]
    null_var_index = np.genfromtxt(null_variance_index_path, delimiter=",", skip_header=0, dtype=int)
    mean_tx = np.genfromtxt(mean_path, delimiter=",", skip_header=0)
    std_tx = np.genfromtxt(std_path, delimiter=",", skip_header=0)
    tosolve_tx = np.genfromtxt(eigenvectors_path, delimiter=",", skip_header=0, dtype=np.complex128)

    return w, thresh, null_var_index, mean_tx, std_tx, tosolve_tx

def PRI_jet_num_split(data):
    """Splitting the dataset according to PRI_jet_num which is the categorical feature in our dataset."""
    tX = rearrange_continuous_categorical_features(data)
    categories = tX[:, -1]
    zeros_index = np.where(categories == 0)[0]
    one_index = np.where(categories == 1)[0]
    two_index = np.where(categories == 2)[0]
    three_index = np.where(categories == 3)[0]
    
    zeros = tX[zeros_index, :]
    ones = tX[one_index, :]
    two = tX[two_index, :]
    three = tX[three_index, :]

    return zeros, ones, two, three, zeros_index, one_index, two_index, three_index

def process_testdata(data, null_var_index, degree, train_mean, train_std, train_eigenvectors):
    """Apply the necessary transformations to be aligned with the training data."""
    data = np.delete(data, null_var_index, axis=1)
    data[np.where(data == -999)] = np.nan
    data = median_imputation(data)
    data = process_data(x = data, degree=degree, pairwise=True, bias=False)
    data = (data - train_mean) / train_std
    data = np.linalg.solve(train_eigenvectors, data.T).T
    data = process_data(x = data, degree=0, pairwise=False, bias=True)
    
    return data

def predict(tx, w, thresh):
    pred = sigmoid(tx@w)
    pred = (pred>thresh)*1
    pred[np.where(pred == 0)] = -1
    
    return pred
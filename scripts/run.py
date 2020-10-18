### Run file

import numpy as np
from proj1_helpers import *
from data_processing import *

from objective_functions import *

def import_parameters(w_path, threshold_path, null_variance_index_path, mean_path, std_path, eigenvectors_path):
    """Importing relevant parameters for predictions."""
    w = np.genfromtxt(w_path, delimiter=",", skip_header=1, dtype=np.complex128)
    thresh = np.genfromtxt(threshold_path, delimiter=",", skip_header=1)[0]
    null_var_index = np.genfromtxt(null_variance_index_path, delimiter=",", skip_header=1, dtype=int)
    mean_tx = np.genfromtxt(mean_path, delimiter=",", skip_header=1)
    std_tx = np.genfromtxt(std_path, delimiter=",", skip_header=1)
    tosolve_tx = np.genfromtxt(eigenvectors_path, delimiter=",", skip_header=1, dtype=np.complex128)

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

# The procedure we use is as follows:
#  1 - We locate the data points that corresponds to PRI_jet_num = 0, 1, 2, 3 and we separate them into four datasets.
# For each of the datasets:
#  2 - We delete the columns whose data are invariant, the same as in the training data.
#  3 - We impute the missing values with the median.
#  4 - We perform polynomial and pairwise interaction augmentation.
#  5 - We scale them using values from the training sets.
#  6 - We express them in an orthogonal basis, the same as in the training data.
#  7 - We add a bias.

# All needed parameters: 

#    - w_i : vectors of parameters for model i.
#    - thresh_i : the decision threshold to map a probability to {0,1} for model i.
#    - null_var_index_i : index that indicates invariant columns for model i.
#    - mean_tx_i : vector of means of features for the model i.
#    - std_tx_i : vector of standard deviations of features for the model i.
#    - tosolve_tx_i : matrix of eigenvectors of features for the model i.

# These parameters define our trained model.

print("Importing parameters of trained models...")
w_0, thresh_0, null_var_index_zero, mean_tx_zeros, std_tx_zeros, tosolve_tx_zeros = import_parameters("data/run/w_0.csv", "data/run/thresh_0.csv", "data/run/null_var_index_zero.csv", "data/run/mean_tx_zeros.csv", "data/run/std_tx_zeros.csv","data/run/tosolve_tx_zeros.csv")
w_1, thresh_1, null_var_index_one, mean_tx_ones, std_tx_ones, tosolve_tx_ones =import_parameters("data/run/w_1.csv", "data/run/thresh_1.csv", "data/run/null_var_index_one.csv", "data/run/mean_tx_ones.csv", "data/run/std_tx_ones.csv","data/run/tosolve_tx_ones.csv")
w_2, thresh_2, null_var_index_two, mean_tx_two, std_tx_two, tosolve_tx_two = import_parameters("data/run/w_2.csv", "data/run/thresh_2.csv", "data/run/null_var_index_two.csv", "data/run/mean_tx_two.csv", "data/run/std_tx_two.csv","data/run/tosolve_tx_two.csv")
w_3, thresh_3, null_var_index_three, mean_tx_three, std_tx_three, tosolve_tx_three = import_parameters("data/run/w_3.csv", "data/run/thresh_3.csv", "data/run/null_var_index_three.csv", "data/run/mean_tx_three.csv", "data/run/std_tx_three.csv","data/run/tosolve_tx_three.csv")


# Predicting

print("Loading testset...")
_, tX_test, ids_test = load_csv_data("data/test.csv")

print("Data processing...")
zeros_test, ones_test, two_test, three_test, zeros_index_test, one_index_test, two_index_test, three_index_test = PRI_jet_num_split(tX_test)

zeros_test = process_testdata(zeros_test, null_var_index_zero, 13, mean_tx_zeros, std_tx_zeros, tosolve_tx_zeros)
ones_test = process_testdata(ones_test, null_var_index_one, 17, mean_tx_ones, std_tx_ones, tosolve_tx_ones)
two_test = process_testdata(two_test, null_var_index_two, 13, mean_tx_two, std_tx_two, tosolve_tx_two)
three_test = process_testdata(three_test, null_var_index_three, 10, mean_tx_three, std_tx_three, tosolve_tx_three)

print("Predicting...")
predictions = _
predictions[zeros_index_test] = predict(zeros_test, w_0, thresh_0)
predictions[one_index_test] = predict(ones_test, w_1, thresh_1)
predictions[two_index_test] = predict(two_test, w_2, thresh_2)
predictions[three_index_test] = predict(three_test, w_3, thresh_3)

create_csv_submission(ids_test, predictions, "run_submission.csv")
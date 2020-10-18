### Run file

import numpy as np
from proj1_helpers import *
from functions import *

def import_parameters(w_path, threshold_path, null_variance_index_path, mean_path, std_path, eigenvectors_path):

    w = np.genfromtxt(w_path, delimiter=",", skip_header=1, dtype=np.complex128)
    thresh = np.genfromtxt(threshold_path, delimiter=",", skip_header=1)[0]
    null_var_index = np.genfromtxt(null_variance_index_path, delimiter=",", skip_header=1, dtype=int)
    mean_tx = np.genfromtxt(mean_path, delimiter=",", skip_header=1)
    std_tx = np.genfromtxt(std_path, delimiter=",", skip_header=1)
    tosolve_tx = np.genfromtxt(eigenvectors_path, delimiter=",", skip_header=1, dtype=np.complex128)

    return w, thresh, null_var_index, mean_tx, std_tx, tosolve_tx


print("Execution started")

print("Importing parameters...")
w_0, thresh_0, null_var_index_zero, mean_tx_zeros, std_tx_zeros, tosolve_tx_zeros = import_parameters("data/run/w_0.csv", "data/run/thresh_0.csv", "data/run/null_var_index_zero.csv", "data/run/mean_tx_zeros.csv", "data/run/std_tx_zeros.csv","data/run/tosolve_tx_zeros.csv")
w_1, thresh_1, null_var_index_one, mean_tx_ones, std_tx_ones, tosolve_tx_ones =import_parameters("data/run/w_1.csv", "data/run/thresh_1.csv", "data/run/null_var_index_one.csv", "data/run/mean_tx_ones.csv", "data/run/std_tx_ones.csv","data/run/tosolve_tx_ones.csv")
w_2, thresh_2, null_var_index_two, mean_tx_two, std_tx_two, tosolve_tx_two = import_parameters("data/run/w_2.csv", "data/run/thresh_2.csv", "data/run/null_var_index_two.csv", "data/run/mean_tx_two.csv", "data/run/std_tx_two.csv","data/run/tosolve_tx_two.csv")
w_3, thresh_3, null_var_index_three, mean_tx_three, std_tx_three, tosolve_tx_three = import_parameters("data/run/w_3.csv", "data/run/thresh_3.csv", "data/run/null_var_index_three.csv", "data/run/mean_tx_three.csv", "data/run/std_tx_three.csv","data/run/tosolve_tx_three.csv")
##Explain what are the parameters and what is the output, the time it takes to create the model


_, tX_test, ids_test = load_csv_data("data/test.csv")
tX_test = rearrange_continuous_categorical_features(tX_test)

categories_test = tX_test[:, -1]
zeros_index_test = np.where(categories_test == 0)[0]
one_index_test = np.where(categories_test == 1)[0]
two_index_test = np.where(categories_test == 2)[0]
three_index_test = np.where(categories_test == 3)[0]

zeros_test = tX_test[zeros_index_test, :]
zeros_test = np.delete(zeros_test, null_var_index_zero, axis=1)
zeros_test[np.where(zeros_test == -999)] = np.nan
zeros_test = median_imputation(zeros_test)
zeros_test = process_data(x = zeros_test, degree=13, pairwise=True, bias=False)
zeros_test = (zeros_test - mean_tx_zeros) / std_tx_zeros
zeros_test = np.linalg.solve(tosolve_tx_zeros, zeros_test.T).T
zeros_test = process_data(x = zeros_test, degree=0, pairwise=False, bias=True)


ones_test = tX_test[one_index_test, :]
ones_test = np.delete(ones_test, null_var_index_one, axis=1)
ones_test[np.where(ones_test == -999)] = np.nan
ones_test = median_imputation(ones_test)
ones_test = process_data(x = ones_test, degree=17, pairwise=True, bias=False)
ones_test = (ones_test - mean_tx_ones) / std_tx_ones
ones_test = np.linalg.solve(tosolve_tx_ones, ones_test.T).T
ones_test = process_data(x = ones_test, degree=0, pairwise=False, bias=True)

two_test = tX_test[two_index_test, :]
two_test = np.delete(two_test, null_var_index_two, axis=1)
two_test[np.where(two_test == -999)] = np.nan
two_test = median_imputation(two_test)
two_test = process_data(x = two_test, degree=13, pairwise=True, bias=False)
two_test = (two_test - mean_tx_two) / std_tx_two
two_test = np.linalg.solve(tosolve_tx_two, two_test.T).T
two_test = process_data(x = two_test, degree=0, pairwise=False, bias=True)

three_test = tX_test[three_index_test, :]
three_test = np.delete(three_test, null_var_index_three, axis=1)
three_test[np.where(three_test == -999)] = np.nan
three_test = median_imputation(three_test)
three_test = process_data(x = three_test, degree=10, pairwise=True, bias=False)
three_test = (three_test - mean_tx_three) / std_tx_three
three_test = np.linalg.solve(tosolve_tx_three, three_test.T).T
three_test = process_data(x = three_test, degree=0, pairwise=False, bias=True)


y_pred_zero = sigmoid(zeros_test@w_0)
y_pred_zero = (y_pred_zero>thresh_0)*1
y_pred_zero[np.where(y_pred_zero == 0)] = -1

y_pred_one = sigmoid(ones_test@w_1)
y_pred_one = (y_pred_one>thresh_1)*1
y_pred_one[np.where(y_pred_one == 0)] = -1

y_pred_two = sigmoid(two_test@w_2)
y_pred_two = (y_pred_two>thresh_2)*1
y_pred_two[np.where(y_pred_two == 0)] = -1

y_pred_three = sigmoid(three_test@w_3)
y_pred_three = (y_pred_three>thresh_3)*1
y_pred_three[np.where(y_pred_three == 0)] = -1

predictions = _
predictions[zeros_index_test] = y_pred_zero
predictions[one_index_test] = y_pred_one
predictions[two_index_test] = y_pred_two
predictions[three_index_test] = y_pred_three

create_csv_submission(ids_test, predictions, "run_submission.csv")
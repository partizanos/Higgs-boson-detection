### Run file

import numpy as np
from proj1_helpers import *
from functions import *

w_0 = np.genfromtxt("data/run/w_0.csv", delimiter=",", skip_header=1, dtype=np.complex128)
thresh_0 = np.genfromtxt("data/run/thresh_0.csv", delimiter=",", skip_header=1)[0]
null_var_index_zero = np.genfromtxt("data/run/null_var_index_zero.csv", delimiter=",", skip_header=1, dtype=int)
mean_tx_zeros = np.genfromtxt("data/run/mean_tx_zeros.csv", delimiter=",", skip_header=1)
std_tx_zeros = np.genfromtxt("data/run/std_tx_zeros.csv", delimiter=",", skip_header=1)
tosolve_tx_zeros = np.genfromtxt("data/run/tosolve_tx_zeros.csv", delimiter=",", skip_header=1, dtype=np.complex128)

w_1 = np.genfromtxt("data/run/w_1.csv", delimiter=",", skip_header=1, dtype=np.complex128)
thresh_1 = np.genfromtxt("data/run/thresh_1.csv", delimiter=",", skip_header=1)[0]
null_var_index_one = np.genfromtxt("data/run/null_var_index_one.csv", delimiter=",", skip_header=1, dtype=int)
mean_tx_ones = np.genfromtxt("data/run/mean_tx_ones.csv", delimiter=",", skip_header=1)
std_tx_ones = np.genfromtxt("data/run/std_tx_ones.csv", delimiter=",", skip_header=1)
tosolve_tx_ones = np.genfromtxt("data/run/tosolve_tx_ones.csv", delimiter=",", skip_header=1, dtype=np.complex128)

w_2 = np.genfromtxt("data/run/w_2.csv", delimiter=",", skip_header=1, dtype=np.complex128)
thresh_2 = np.genfromtxt("data/run/thresh_2.csv", delimiter=",", skip_header=1)[0]
null_var_index_two = np.genfromtxt("data/run/null_var_index_two.csv", delimiter=",", skip_header=1, dtype=int)
mean_tx_two = np.genfromtxt("data/run/mean_tx_two.csv", delimiter=",", skip_header=1)
std_tx_two = np.genfromtxt("data/run/std_tx_two.csv", delimiter=",", skip_header=1)
tosolve_tx_two = np.genfromtxt("data/run/tosolve_tx_two.csv", delimiter=",", skip_header=1, dtype=np.complex128)

w_3 = np.genfromtxt("data/run/w_3.csv", delimiter=",", skip_header=1)
thresh_3 = np.genfromtxt("data/run/thresh_3.csv", delimiter=",", skip_header=1)[0]
null_var_index_three = np.genfromtxt("data/run/null_var_index_three.csv", delimiter=",", skip_header=1, dtype=int)
mean_tx_three = np.genfromtxt("data/run/mean_tx_three.csv", delimiter=",", skip_header=1)
std_tx_three = np.genfromtxt("data/run/std_tx_three.csv", delimiter=",", skip_header=1)
tosolve_tx_three = np.genfromtxt("data/run/tosolve_tx_three.csv", delimiter=",", skip_header=1)


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
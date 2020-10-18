import numpy as np

from extra_helpers import *

def build_poly(x, degree, max_categories=10):
    """Polynomial basis augmentation on continuous features.

       - Ignoring the 0-monomial term, i.e use add_bias() function.
       - The 1-monomial term is equivalent to copy the original data. Hence, degree > 1."""

    # Identifying continuous and discrete features.
    cat_index = is_cat(x, max_categories)
    categorical_variables = x[:, cat_index]
    continuous_variables = x[:, ~cat_index]

    # Doing simple polynomial basis augmentation only on continuous features.
    augmented_x = continuous_variables
    dim2 = degree-1
    if degree > 1:
        for j, i in enumerate(range(2, degree+1)):
            augmented_x = np.c_[augmented_x, np.power(continuous_variables, i)]
            print(
                f"Polynomial augmentation progress : {round(((j+1)/dim2)*100, 2)}%                 \r", end="")

    return np.c_[augmented_x, categorical_variables]

def mean_imputation(na_data):
    """Imputing missing values based on feature means."""
    _, clean_data = row_na_omit(np.zeros((na_data.shape[0], 1)), na_data)
    mean = np.mean(clean_data, axis=0)
    for i in range(na_data.shape[1]):
        na_data[:, i] = np.nan_to_num(na_data[:, i], nan=mean[i])

    return na_data

def median_imputation(na_data):
    """Imputing missing values based on feature medians."""
    _, clean_data = row_na_omit(np.zeros((na_data.shape[0], 1)), na_data)
    median = np.median(clean_data, axis=0)
    for i in range(na_data.shape[1]):
        na_data[:, i] = np.nan_to_num(na_data[:, i], nan=median[i])

    return na_data

def stochastic_nearest_neighbor_imputation(na_data, neighbors=10, length=100):
    """Imputing missing values based on means off k nearest neighbors."""
    _, clean_data = row_na_omit(np.zeros((na_data.shape[0], 1)), na_data)
    n_iters = na_data.shape[0]
    for i in range(n_iters):
        print(f"{round(((i+1)/n_iters)*100, 2)}%                \r", end="")
        condition = np.isnan(na_data[i, :])
        if len(condition) > 0:
            sample = random_sample(clean_data, length)
            index = np.where(condition)
            candidate = np.delete(na_data[i], index)
            neighborhood = np.delete(sample, index, axis=1)
            distances = pairwise(candidate.reshape(
                (1, len(candidate))), neighborhood)
            nearest_index = np.argsort(distances)[0][:neighbors]
            na_data[i, index] = np.mean(sample[nearest_index], axis=0)[index]

    return(na_data)


def pairwise_interaction(x, max_categories=10):
    """Pairwise interactions/product between continuous variables."""
    # Identifying continuous and discrete features.
    cat_index = is_cat(x, max_categories)
    categorical_variables = x[:, cat_index]
    continuous_variables = x[:, ~cat_index]

    augmented_x = continuous_variables
    dim2 = continuous_variables.shape[1]
    for j in range(dim2):
        augmented_x = np.c_[augmented_x, continuous_variables[:, j].reshape(
            (continuous_variables.shape[0], 1)) * continuous_variables[:, j+1:]]
        print(
            f"Pairwise interaction progress : {round(((j+1)/dim2)*100, 2)}%                 \r", end="")

    return np.c_[augmented_x, categorical_variables]

def trigonometric_augmentation(x, max_categories=10):
    """Cos/sin augmentation for continuous variables."""
    # Identifying continuous and discrete features.
    cat_index = is_cat(x, max_categories)
    categorical_variables = x[:, cat_index]
    continuous_variables = x[:, ~cat_index]

    augmented_x = continuous_variables
    augmented_x = np.c_[augmented_x, np.cos(
        continuous_variables), np.sin(continuous_variables)]
    print(f"Trigonometric augmentation : ✔                 \r", end="")

    return np.c_[augmented_x, categorical_variables]


def logabs1_augmentation(x, max_categories=10):
    # Identifying continuous and discrete features.
    cat_index = is_cat(x, max_categories)
    categorical_variables = x[:, cat_index]
    continuous_variables = x[:, ~cat_index]

    augmented_x = continuous_variables
    augmented_x = np.c_[augmented_x, np.log(
        np.abs(continuous_variables)+1)]
    print(f"logabs1 augmentation : ✔                 \r", end="")

    return np.c_[augmented_x, categorical_variables]

def exp_augmentation(x, max_categories=10):
    # Identifying continuous and discrete features.
    cat_index = is_cat(x, max_categories)
    categorical_variables = x[:, cat_index]
    continuous_variables = x[:, ~cat_index]

    augmented_x = continuous_variables
    augmented_x = np.c_[augmented_x, np.exp(continuous_variables)]
    print(f"exp augmentation : ✔                 \r", end="")

    return np.c_[augmented_x, categorical_variables]

def add_bias(x):
    """Adding a bias/intercept to a data matrix."""
    res = np.c_[np.ones((x.shape[0], 1)), x]
    print(f"Bias : ✔                                  \r", end="")
    return res

def process_data(x, degree=0, bias=False, pairwise=False, trigonometric_functions=False, logabs1=False, exp=False, create_dummies=False, max_categories=10):
    """Polynomial basis augmentation, pairwise interactions, trigonometric transformations,
       log/exp transformations and categorical variable to dummy variables."""

    # Identifying continuous and discrete features.
    cat_index = is_cat(x, max_categories)
    categorical_variables = x[:, cat_index]
    continuous_variables = x[:, ~cat_index]

    # Storing the shape of continuous_variables to avoid repeating terms.
    dim1, dim2 = continuous_variables.shape

    if degree > 1:
        print("")
        poly = build_poly(continuous_variables, degree)[:, dim2:]
    else:
        poly = np.zeros((dim1, 1))

    # Doing pairwise product only on continuous features.
    if pairwise:
        print("")
        inter = pairwise_interaction(continuous_variables)[:, dim2:]
    else:
        inter = np.zeros((dim1, 1))

    # Adding trigonometric transformations.
    if trigonometric_functions:
        print("")
        trigo = trigonometric_augmentation(continuous_variables)[:, dim2:]
    else:
        trigo = np.zeros((dim1, 1))

    # Adding logarithm of absolute values.
    if logabs1:
        print("")
        ln = logabs1_augmentation(continuous_variables)[:, dim2:]
    else:
        ln = np.zeros((dim1, 1))

    # Adding exponential transformations.
    if exp:
        print("")
        ex = exp_augmentation(continuous_variables)[:, dim2:]
    else:
        ex = np.zeros((dim1, 1))

    # Deleting constant variables, i.e. either useless or artificially created when if condition was not satisfied.
    augmented_x = np.c_[continuous_variables, poly, inter, trigo, ln, ex]
    null_var_index = np.where(np.std(augmented_x, axis=0) == 0)[0]
    augmented_x = np.delete(augmented_x, null_var_index, axis=1)

    # Adding categorical variables as dummy variables.
    if create_dummies:
        print("")
        dim2 = categorical_variables.shape[1]
        for j in range(dim2):
            unique = list(set(np.squeeze(categorical_variables[:, j])))
            result = [i == unique for i in np.squeeze(
                categorical_variables[:, j])]
            augmented_x = np.c_[augmented_x, np.squeeze(result)*1]
            print(
                f"Dummy variables progress : {round(((j+1)/dim2)*100, 2)}%                 \r", end="")
    else:
        augmented_x = np.c_[augmented_x, categorical_variables]

    if bias:
        print("")
        augmented_x = add_bias(augmented_x)

    return augmented_x

def orthogonal_basis(x):
    """Orthogonal change of basis."""
    cov = np.cov(x, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    x_orth = np.linalg.solve(eigenvectors, x.T).T

    return x_orth, eigenvectors


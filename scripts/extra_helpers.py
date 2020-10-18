import numpy as np

def is_cat(x, max_categories=10):
    """Check if an array is categorical."""
    if sum(sum(np.isnan(x))):
        raise ValueError('Missing values (np.nan) not allowed.')
    boolean_index = list([])
    if x.shape == (len(x),):
        if len(set(x)) < max_categories:
            boolean = True
        else:
            boolean = False
            boolean_index.append(boolean)
    else:
        for i in range(x.shape[1]):
            if len(set(x[:, i])) < max_categories:
                boolean = True
            else:
                boolean = False
            boolean_index.append(boolean)
    return np.array(boolean_index)

def rearrange_continuous_categorical_features(x, max_categories=10):
    """Rearranging features such that continuous variable are separated from categorical variables."""
    # Identifying continuous and discrete features.
    cat_index = is_cat(x, max_categories)
    categorical_variables = x[:, cat_index]
    continuous_variables = x[:, ~cat_index]

    # Rearranging.
    x = np.c_[continuous_variables, categorical_variables]

    return x

def gaussian_scaling(x, max_categories=10):
    """Scaling data by subtracting the mean and dividing by the standard deviation."""
    # For loops are used because of lack of memory for huge datasets.

    # Identifying continuous and discrete features.
    cat_index = is_cat(x, max_categories)
    categorical_variables = x[:, cat_index]
    continuous_variables = x[:, ~cat_index]

    # Scale only continuous features.
    mean_ = list()
    std_ = list()
    for i in range(continuous_variables.shape[1]):
        a = np.mean(continuous_variables[:, i], axis=0)
        b = np.std(continuous_variables[:, i], axis=0)
        mean_.append(a)
        std_.append(b)
        continuous_variables[:, i] = (continuous_variables[:, i] - a) / b

    # Setting categorical means and std to 0 and 1.
    mean_ = np.array(mean_)
    std_ = np.array(std_)
    x_scaled = np.c_[continuous_variables, categorical_variables]
    mean_ = np.concatenate((mean_, np.zeros(sum(cat_index*1))))
    std_ = np.concatenate((std_, np.ones(sum(cat_index*1))))

    return x_scaled, mean_, std_

def col_na_omit(x, tol=1):
    """Delete columns of a dataset with a higher proportion of missing values than tol. 0⩽tol⩽1."""
    index = sum(np.isnan(x))/(x.shape[0]) > tol
    x = x[:, ~index]

    return x

def row_na_omit(y, x):
    """Delete all rows of a dataset containing at least one missing value."""
    index = np.isnan(x).any(axis=1)
    x_no_na = x[~index]
    y_no_na = y[~index]

    return y_no_na, x_no_na

def remove_outliers(y, x, quantile=1.96):
    """Delete all rows of a dataset having at least one value outside their respective feature confidence interval (CI).
       CI is symmetric and computed based on Gaussian assumption. Can be used on the training set, but not on the test set."""
    _, clean_data = row_na_omit(y, x)
    m = np.mean(clean_data, axis=0)
    s = np.std(clean_data, axis=0)
    lower = m - quantile * s
    upper = m + quantile * s

    for i in range(clean_data.shape[1]):

        index = clean_data[:, i] < lower[i]
        clean_data = np.delete(clean_data, index*1, axis=0)

        index = x[:, i] < lower[i]
        x = np.delete(x, index*1, axis=0)
        y = np.delete(y, index*1, axis=0)

        index = clean_data[:, i] > upper[i]
        clean_data = np.delete(clean_data, index*1, axis=0)

        index = x[:, i] > upper[i]
        x = np.delete(x, index*1, axis=0)
        y = np.delete(y, index*1, axis=0)

    return y, x

def skewness(x):
    """Compute the skewness of all features."""
    mean_x = np.mean(x, axis=0)
    std_x = np.std(x, axis=0)
    skew = np.mean(((x - mean_x)/std_x)**3, axis=0)

    return skew

def split_data(y, x, ratio, seed=1):
    """split the dataset based on the split ratio."""
    np.random.seed(seed)
    num_row = len(y)
    indices = np.random.permutation(num_row)
    index_split = int(np.floor(ratio * num_row))
    index_train = indices[:index_split]
    index_validation = indices[index_split:]
    x_train = x[index_train]
    x_validation = x[index_validation]
    y_train = y[index_train]
    y_validation = y[index_validation]

    return x_train, x_validation, y_train, y_validation

def pairwise(p, q):
    """Computing pairwise Euclidean distances. Used for stochastic nearest neighbor imputation."""

    return np.sqrt(np.sum((p[:, np.newaxis, :]-q[np.newaxis, :, :])**2, axis=2))

def random_sample(x, length):
    """Returns a random sample of x for a given length. No seed involved."""
    num_row = x.shape[0]
    indices = np.random.permutation(num_row)
    sample = x[indices][:length]

    return sample

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]

def choose(n, k):
    """Binomial coefficient."""
    return int(np.math.factorial(n)/(np.math.factorial(k)*np.math.factorial(n-k)))

def threshold(y, fitted_probabilities, step=0.01):
    """find the best threshold for classification"""
    candidates = np.arange(0.2, 0.8, step)
    thresholds = list([])
    accuracies = list([])
    for i in candidates:
        prediction = (fitted_probabilities > i)*1
        accuracy = 1 - sum(np.abs(prediction - y))/len(y)
        thresholds.append(i)
        accuracies.append(accuracy)
    index = accuracies.index(max(accuracies))

    return thresholds[index]

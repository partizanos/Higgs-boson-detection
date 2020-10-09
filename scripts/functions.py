import numpy as np


def is_cat(x, max_categories=10):
    """Check if an array is categorical"""
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
    # Identifying continuous and discrete features.
    cat_index = is_cat(x, max_categories)
    categorical_variables = x[:, cat_index]
    continuous_variables = x[:, ~cat_index]

    # Scale only continuous features.
    mean_ = np.mean(continuous_variables, axis=0)
    continuous_variables_scaled = continuous_variables - mean_
    std_ = np.std(continuous_variables, axis=0)
    continuous_variables_scaled = continuous_variables_scaled / std_

    # Setting categorical means and std to 0 and 1.
    x_scaled = np.c_[continuous_variables_scaled, categorical_variables]
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


def remove_outliers(y, x, quantile=3):
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


def sigmoid(x):
    """Sigmoid function."""

    return 1.0 / (1 + np.exp(-x))


def logit_loss(y, tx, w):
    """Logistic log-likelihood."""
    proba = sigmoid(tx.dot(w))
    loss = (-1/len(y))*(y.T.dot(np.log(proba)) +
                        (1 - y).T.dot(np.log(1 - proba)))

    return loss


def logit_gradient(y, tx, w):
    """Logistic log-likelihood gradient."""
    proba = sigmoid(tx.dot(w))
    grad = tx.T.dot(proba - y)

    return (1/len(y))*grad


def least_squares(y, tx):
    """OLS."""

    return np.linalg.solve(tx.T.dot(tx), tx.T.dot(y))


def logit_hessian(y, tx, w):
    """Logistic log-likelihood hessian."""
    proba = sigmoid(tx.dot(w))
    diag = np.multiply(proba, (1-proba))
    X_tilde = tx * diag.reshape((len(diag), 1))

    return (1/len(y))*tx.T.dot(X_tilde)


def logistic_newton_descent(y, tx, w, max_iters, eps=1e-4, w_start_OLS=True):
    """Newton descent."""
    if w_start_OLS:
        w = least_squares(y, tx)
    grad = logit_gradient(y, tx, w)
    hess = logit_hessian(y, tx, w)
    norm = np.linalg.norm(grad, np.inf)
    counter = 0
    while norm > eps:
        print(f"Gradient norm = {round(norm, 7)}                 \r", end="")
        counter += 1
        grad = logit_gradient(y, tx, w)
        hess = logit_hessian(y, tx, w)
        norm = np.linalg.norm(grad, np.inf)

        try:
            w1 = w - np.linalg.solve(hess, grad)
            if np.linalg.norm(logit_gradient(y, tx, w1), np.inf) < norm:
                w = w1
            else:
                print(f"Maximum progress reached until divergence.                        ")
                break
        except:
            print(f"Singular Hessian.                        ")
            break

        if counter == max_iters:
            print(f"max_iters reached.                        ")
            break

    return logit_loss(y, tx, w), w, norm


def logistic_gradient_descent(y, tx, w, max_iters, gamma, eps=1e-4, w_start_OLS=True):
    """logistic gradient descent."""
    if w_start_OLS:
        w = least_squares(y, tx)
    grad = logit_gradient(y, tx, w)
    norm = np.linalg.norm(grad, np.inf)
    counter = 0
    while norm > eps:
        counter += 1
        grad = logit_gradient(y, tx, w)
        w -= gamma * grad
        norm = np.linalg.norm(grad, np.inf)
        print(f"Gradient norm = {round(norm, 7)}                 \r", end="")
        if counter == max_iters:
            print(f"max_iters reached.                        ")
            break

    return logit_loss(y, tx, w), w, norm


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


def logistic_stochastic_gradient_descent(y, tx, w, max_iters, gamma, batch_size=1, w_start_OLS=True):
    """logistic stochastic gradient descent."""
    if w_start_OLS:
        w = least_squares(y, tx)
    grad = logit_gradient(y, tx, w)
    norm = np.linalg.norm(grad, np.inf)
    counter = 0
    
    for y_batch, tx_batch in batch_iter(y, tx, batch_size=batch_size, num_batches=max_iters):
        counter += 1
        grad = logit_gradient(y_batch, tx_batch, w)
        w -= gamma * grad
        norm = np.linalg.norm(grad, np.inf)
        print(f"Progress : {round((counter/max_iters)*100, 2)}%                 \r", end="")

    return logit_loss(y, tx, w), w, norm


def choose(n, k):
    """Binomial coefficient."""
    return int(np.math.factorial(n)/(np.math.factorial(k)*np.math.factorial(n-k)))


def build_poly(x, degree, pairwise_interaction=True, intercept=False, max_categories=10):
    """Build polynomial basis augmentation, pairwise interactions and an intercept."""
    # Identifying continuous and discrete features.
    cat_index = is_cat(x, max_categories)
    categorical_variables = x[:, cat_index]
    continuous_variables = x[:, ~cat_index]

    # Doing simple polynomial basis augmentation only on continuous features.
    augmented_x = continuous_variables
    if degree > 1:
        for i in range(2, degree+1):
            augmented_x = np.c_[augmented_x, np.power(continuous_variables, i)]

    # Doing pairwise product only on continuous features.
    if pairwise_interaction:
        nb_dim = continuous_variables.shape[1]
        # Number of interactions to follow progress.
        nb_pairs = choose(nb_dim, 2)
        counter = 0
        for j in range(nb_dim):
            for k in range(nb_dim):
                if j >= k:
                    continue  # Ensure uniqueness of product.
                else:
                    augmented_x = np.c_[augmented_x, np.multiply(
                        continuous_variables[:, j], continuous_variables[:, k])]
                    counter += 1
                    print(
                        f"Progress : {round((counter/nb_pairs)*100, 2)}%                 \r", end="")

    # Adding an intercept.
    if intercept:
        inter = np.ones((x.shape[0], 1))
        augmented_x = np.c_[np.ones((x.shape[0], 1)), augmented_x]

    # Adding categorical variables.
    augmented_x = np.c_[augmented_x, categorical_variables]

    return augmented_x


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


def eigen(train, var=99.99999999):
    """Eigen-decomposition for a minimum variance explained (%).
       Returns the basis-changed matrix and retained eigenvectors for further
       transformation."""
    A = np.cov(train, rowvar=False)
    eigenValues, eigenVectors = np.linalg.eig(A)
    idx = np.argsort(-eigenValues)
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:, idx]
    candidates = (np.cumsum(eigenValues)/np.sum(eigenValues))
    index = np.where(candidates >= (var/100))[0][0]
    sub_linear_space = eigenVectors.real[:, :(index+1)]
    train = train.dot(sub_linear_space)
    return train, sub_linear_space
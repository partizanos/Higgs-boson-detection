import numpy as np


def riemann_approximation_gaussian_cdf(b, N=100000):
    """Riemann approximation of gaussian distribution."""
    inte = 0
    a = -50
    for i in range(N):
        x = a+(b-a)*i/N
        inte += (np.exp(-x**2/2)*(b-a)/N)/np.sqrt(2*np.pi)
    return inte


def gaussian_test(y, feature):
    """Statistical test to quantify feature importance."""
    boson_index = np.where(y == 1)[0]

    boson_feature = feature[boson_index]
    negative_feature = feature[~boson_index]

    boson_mean = np.mean(boson_feature)
    negative_mean = np.mean(negative_feature)

    boson_var = (1/len(boson_feature))*np.var(boson_feature)
    negative_var = (1/len(negative_feature))*np.var(negative_feature)

    t_var = boson_var + negative_var
    t_mean = boson_mean - negative_mean
    test = t_mean/np.sqrt(t_var)

    # Under H0, test ~ N(0,1)
    return 2*(1-riemann_approximation_gaussian_cdf(abs(test)))


def most_important_features_index(y, x, max_features=8):
    """Returns an index of features by importance up to max_features."""
    x = x[:, ~is_cat(x)]
    p_values = list()
    for i in range(x.shape[1]):
        p_values.append(gaussian_test(y, x[:, i]))
    return np.argsort(np.array(p_values))[:max_features]


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


def logit_loss(y, tx, w, lambda_=0):
    """Logistic log-likelihood."""
    proba = sigmoid(tx.dot(w))
    loss = (-1/len(y))*(y.T.dot(np.log(proba)) +
                        (1 - y).T.dot(np.log(1 - proba))) + 2*lambda_*np.linalg.norm(w)

    return loss


def logit_gradient(y, tx, w, lambda_=0):
    """Logistic log-likelihood gradient."""
    proba = sigmoid(tx.dot(w))
    grad = (1/len(y))*tx.T.dot(proba - y) + 2*lambda_*w # /np.linalg.norm(w)

    return grad


def least_squares(y, tx, lambda_=0):
    """OLS."""

    return np.linalg.solve(tx.T.dot(tx)+2*lambda_*np.eye(tx.shape[1]), tx.T.dot(y))


def logit_hessian(y, tx, w, lambda_=0):
    """Logistic log-likelihood hessian."""
    proba = sigmoid(tx.dot(w))
    diag = np.multiply(proba, (1-proba))
    X_tilde = tx * diag.reshape((len(diag), 1))
    # t0 = 2*lambda_
    # t1 = np.linalg.norm(w)

    return (1/len(y))*(tx.T.dot(X_tilde)) + 2 * lambda_ * np.eye(tx.shape[1])# +t0/t1*np.eye(len(w))-t0/(t1**3)*np.kron(w, w).reshape(len(w), len(w))


def logistic_newton_descent(y, tx, w, max_iters, lambda_=0, eps=1e-4, w_start_OLS=False):
    """Newton descent."""
    if w_start_OLS:
        try:
            w = least_squares(y, tx, lambda_)
        except:
            w = np.zeros(tx.shape[1])

    grad = logit_gradient(y, tx, w, lambda_)
    hess = logit_hessian(y, tx, w, lambda_)
    norm = np.linalg.norm(grad, np.inf)
    counter = 0
    while norm > eps:
        print(f"Gradient norm = {round(norm, 7)}                 \r", end="")
        counter += 1
        grad = logit_gradient(y, tx, w, lambda_)
        hess = logit_hessian(y, tx, w, lambda_)
        norm = np.linalg.norm(grad, np.inf)

        try:
            w1 = w - np.linalg.solve(hess, grad)
            if np.linalg.norm(logit_gradient(y, tx, w1, lambda_), np.inf) < norm:
                w = w1
            else:
                #print(f"Maximum progress reached until divergence.                        ")
                break
        except:
            #print(f"Singular Hessian.                        ")
            break

        if counter == max_iters:
            #print(f"max_iters reached.                        ")
            break

    return logit_loss(y, tx, w, lambda_), w, norm


def logistic_gradient_descent(y, tx, w, max_iters, gamma, lambda_=0, eps=1e-4, w_start_OLS=False):
    """logistic gradient descent."""
    if w_start_OLS:
        w = least_squares(y, tx, lambda_)
    grad = logit_gradient(y, tx, w, lambda_)
    norm = np.linalg.norm(grad, np.inf)
    counter = 0
    while norm > eps:
        counter += 1
        grad = logit_gradient(y, tx, w, lambda_)
        w -= gamma * grad
        norm = np.linalg.norm(grad, np.inf)
        print(f"Gradient norm = {round(norm, 7)}                 \r", end="")
        if counter == max_iters:
            #print(f"max_iters reached.                        ")
            break

    return logit_loss(y, tx, w, lambda_), w, norm


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


def logistic_stochastic_gradient_descent(y, tx, w, max_iters, gamma, lambda_=0, batch_size=1, w_start_OLS=True):
    """logistic stochastic gradient descent."""
    if w_start_OLS:
        w = least_squares(y, tx, lambda_)
    grad = logit_gradient(y, tx, w, lambda_)
    norm = np.linalg.norm(grad, np.inf)
    counter = 0

    for y_batch, tx_batch in batch_iter(y, tx, batch_size=batch_size, num_batches=max_iters):
        counter += 1
        grad = logit_gradient(y_batch, tx_batch, w, lambda_)
        w -= gamma * grad
        norm = np.linalg.norm(grad, np.inf)
        print(
            f"Progress : {round((counter/max_iters)*100, 2)}%                 \r", end="")

    return logit_loss(y, tx, w, lambda_), w, norm


def choose(n, k):
    """Binomial coefficient."""
    return int(np.math.factorial(n)/(np.math.factorial(k)*np.math.factorial(n-k)))


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


def eigen(x, var=100):
    """Eigen-decomposition for a minimum variance explained (%).
       Returns the basis-changed matrix and retained eigenvectors for further
       transformation."""
    A = np.cov(x, rowvar=False)
    eigenValues, eigenVectors = np.linalg.eig(A)
    idx = np.argsort(-eigenValues)
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:, idx]
    if var == 100:
        sub_linear_space = eigenVectors.real
        x = x.dot(sub_linear_space)
    else:
        candidates = (np.cumsum(eigenValues)/np.sum(eigenValues))
        index = np.where(candidates >= (var/100))[0][0]
        sub_linear_space = eigenVectors.real[:, :(index+1)]
        x = x.dot(sub_linear_space)

    return x, sub_linear_space


def orthogonal_basis(x):
    """Orthogonal change of basis."""
    cov = np.cov(x, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    x_orth = np.linalg.solve(eigenvectors, x.T).T

    return x_orth, eigenvectors


def bootstrap_validation(y, x, repeats=10, splitting_ratio=0.8, start_OLS=True, lambda_=0):
    """Returns bootstrap validation accuracies."""
    Accuracies = list()
    random_seeds = np.random.randint(1, 999, repeats)

    for counter, bootstrap in enumerate(random_seeds):
        x_train, x_validation, y_train, y_validation = split_data(
            y, x, splitting_ratio, seed=bootstrap)
        newton_loss, w, newton_grad_norm = logistic_newton_descent(y_train,
                                                                   x_train,
                                                                   w=np.zeros(
                                                                       x_train.shape[1]),
                                                                   lambda_=lambda_,
                                                                   max_iters=100,
                                                                   eps=1e-6,
                                                                   w_start_OLS=start_OLS)
        GD_loss, w, GD_grad_norm = logistic_gradient_descent(y_train,
                                                             x_train,
                                                             w=w,
                                                             max_iters=1000,
                                                             lambda_=lambda_,
                                                             gamma=0.05,
                                                             eps=1e-4,
                                                             w_start_OLS=False)

        thresh = threshold(y_train, sigmoid(x_train@w))
        pred = (sigmoid(x_validation@w) > thresh)*1
        accuracy = 1 - sum(np.abs(pred - y_validation))/len(y_validation)

        print(
            f"Bootstrap {counter+1}/{len(random_seeds)} --- seed : {bootstrap} --- Validation accuracy : {round(accuracy*100,3)}%")

        Accuracies.append(accuracy)

    mean_ = np.mean(Accuracies)
    median_ = np.median(Accuracies)
    std_ = np.std(Accuracies)
    print("\n", "\n", f"Mean validation accuracy = {mean_}", "\n",
          f"Median validation accuracy = {median_}", "\n", f"Std validation accuracy = {std_}")
    return np.array(Accuracies), mean_, median_, std_, random_seeds


def cross_validation(y, x, k_fold, seed=42, lambda_=0, start_OLS=True):
    """Returns k-fold cross validation validation accuracies."""
    # Creating k_indices
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    k_indices = np.array(k_indices)

    # Store accuracies
    Accuracies = list()

    # Compute CV
    for j in range(k_indices.shape[0]):
        # 1
        train_ind = np.squeeze(
            k_indices[~(np.arange(k_indices.shape[0]) == j)]).reshape(-k_fold+1)
        test_ind = k_indices[j]

        # 2
        x_test = x[test_ind, :]
        y_test = y[test_ind]
        x_train = x[train_ind, :]
        y_train = y[train_ind]

        newton_loss, w, newton_grad_norm = logistic_newton_descent(y_train,
                                                                   x_train,
                                                                   w=np.zeros(
                                                                       x_train.shape[1]),
                                                                   lambda_=lambda_,
                                                                   max_iters=1000,
                                                                   eps=1e-10,
                                                                   w_start_OLS=start_OLS)
        GD_loss, w, GD_grad_norm = logistic_gradient_descent(y_train,
                                                             x_train,
                                                             w=w,
                                                             max_iters=1000,
                                                             lambda_=lambda_,
                                                             gamma=0.05,
                                                             eps=1e-4,
                                                             w_start_OLS=False)

        thresh = threshold(y_train, sigmoid(x_train@w))
        pred = (sigmoid(x_test@w) > thresh)*1
        accuracy = 1 - sum(np.abs(pred - y_test))/len(y_test)

        print(
            f"CV {j+1}/{k_fold} --- Validation accuracy : {round(accuracy*100,3)}%")

        Accuracies.append(accuracy)

    mean_ = np.mean(Accuracies)
    median_ = np.median(Accuracies)
    std_ = np.std(Accuracies)
    print("\n", "\n", f"Mean validation accuracy = {mean_}", "\n",
          f"Median validation accuracy = {median_}", "\n", f"Std validation accuracy = {std_}")

    return np.array(Accuracies), mean_, median_, std_
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


def most_important_features_index(y, x, max_features=8):
    """Returns an index of features by importance up to max_features."""
    x = x[:, ~is_cat(x)]
    p_values = list()
    for i in range(x.shape[1]):
        p_values.append(gaussian_test(y, x[:, i]))
    return np.argsort(np.array(p_values))[:max_features]


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

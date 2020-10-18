import numpy as np

from objective_functions import *
from extra_helpers import *

def least_squares_GD(y, tx, initial_w, max_iter, gamma):
    """Calculate the loss and the w vector produced by the gradient descent algorithm"""
    w = initial_w
    n = len(y)
    for n_iter in range(max_iter): 
        gradient = compute_gradient(y, tx, w)

        w = w - gamma*gradient 
        txw = tx.dot(w)

        loss = np.sum(np.square(txw-y))/n
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iter - 1, l=loss, w0=w[0], w1=w[1]))
    return loss, w


def least_squares_SGD(y, tx, initial_w, max_iter, gamma):
    w = initial_w
    n = len(y)
    for n_iter in range(max_iter):
        entry_num = np.random.randint(n)
        y_entry = y[entry_num]
        tx_entry = tx[entry_num]
        
        gradient = compute_stochastic_gradient(y_entry, tx_entry, w)

        w = w - gamma*gradient 
        txw = tx.dot(w)
        
        #decide on loss at every iteration
        loss = np.sum(np.square(txw-y))/n
        #remove print later
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iter - 1, l=loss, w0=w[0], w1=w[1]))
    return loss, w

def least_squares(y, tx):
    A = tx.T @ tx
    b = tx.T @ y 
    w = np.linalg.solve(A,b)
    
    loss = compute_mse_loss(y, tx, w)
    return loss, w 


def ridge_regression(y, tx, lambda_):
    A = tx.T @ tx + 2*lambda_*np.eye(tx.shape[1])
    b = tx.T @ y 
    w = np.linalg.solve(A,b)
    
    loss = compute_mse_loss(y, tx, w)
    return loss, w

def logistic_regression(y, tx, initial_w, max_iter, gamma):
    return loss, w

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iter, gamma):
    return loss, w


#### Extended versions of regression algorithms

def logistic_gradient_descent(y, tx, w, max_iters, gamma, lambda_=0, eps=1e-4, w_start_OLS=False):
    """logistic gradient descent. Additional provided arguments: eps (stopping value for the infinity norm) and w_start_OLS (starting can be OLS)"""
    if w_start_OLS:
        try:
            loss, w = ridge_regression(y, tx, lambda_)
        except:
            pass
        
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


def logistic_stochastic_gradient_descent(y, tx, w, max_iters, gamma, lambda_=0, batch_size=1, w_start_OLS=True):
    """logistic stochastic gradient descent."""
    if w_start_OLS:
        try:
            loss, w = ridge_regression(y, tx, lambda_)
        except:
            pass
        
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


def logistic_newton_descent(y, tx, w, max_iters, lambda_=0, eps=1e-4, w_start_OLS=False):
    """Newton descent."""
    if w_start_OLS:
        try:
            loss, w = ridge_regression(y, tx, lambda_)
        except:
            pass

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


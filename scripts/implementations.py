import numpy as np
import random

def compute_gradient(y, tx, w):
    """Compute the gradient."""
    n = len(y)
    txw = tx.dot(w)
    gradient = (1/n)*(tx.T.dot((txw - y)))
    
    return gradient

def compute_stochastic_gradient(y, tx, w):
    """Compute the gradient."""
    txw = tx.dot(w)
    gradient = tx.T.dot((txw - y))
    
    return gradient

def compute_mse_loss(y, tx, w):
    """Calculate the MSE loss."""

    txw = tx.dot(w)
    mse = np.square(np.subtract(txw, y)).mean()
    
    return mse


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
        entry_num = random.randint(0,n-1)
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
    pass

def logistic_regression(y, tx, initial_w, max_iter, gamma):
    pass

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iter, gamma):
    pass
import numpy as np

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

def sigmoid(x):
    """Sigmoid function."""

    return 1.0 / (1 + np.exp(-x))

def logit_loss(y, tx, w):
    """Logistic log-likelihood."""
    proba = sigmoid(tx.dot(w))
    loss = (-1/len(y))*(y.T.dot(np.log(proba)) +
                        (1 - y).T.dot(np.log(1 - proba)))

    return loss

def logit_gradient(y, tx, w, lambda_=0):
    """Logistic log-likelihood gradient."""
    proba = sigmoid(tx.dot(w))
    grad = (1/len(y))*tx.T.dot(proba - y) + 2*lambda_*w 

    return grad

def logit_hessian(y, tx, w, lambda_=0):
    """Logistic log-likelihood hessian."""
    proba = sigmoid(tx.dot(w))
    diag = np.multiply(proba, (1-proba))
    X_tilde = tx * diag.reshape((len(diag), 1))

    return (1/len(y))*(tx.T.dot(X_tilde)) + 2 * lambda_ * np.eye(tx.shape[1])
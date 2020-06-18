import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype


def normalize(X):  # creating standard variables here (u-x)/sigma
    if isinstance(X, pd.DataFrame):
        for c in X.columns:
            if is_numeric_dtype(X[c]):
                u = np.mean(X[c])
                s = np.std(X[c])
                X[c] = (X[c] - u) / s
        return
    for j in range(X.shape[1]):
        u = np.mean(X[:, j])
        s = np.std(X[:, j])
        X[:, j] = (X[:, j] - u) / s


def MSE(X, y, B, lmbda):
    # Finished. MSE is a value, changes with B, so use np.sum rather than np.multiply
    error = y - np.dot(X, B)
    return np.sum(np.multiply(error, error))


def loss_gradient(X, y, B, lmbda):
    # Finished.
    return -np.dot(X.T, y-np.dot(X, B))


def loss_ridge(X, y, B, lmbda):
    # Finished. Loss is a value
    B[0] = np.mean(y)
    error = y - np.dot(X, B)
    return np.sum(np.multiply(error, error)) + lmbda*np.sum(np.multiply(B,B))


def loss_gradient_ridge(X, y, B, lmbda):
    # Finished.
    return -np.dot(X.T, (y-np.dot(X, B))) + np.dot(lmbda, B)


def sigmoid(z):
    # Finished
    return 1/(1 + np.exp(-z))


def log_likelihood(X, y, B, lmbda):
    return -np.sum(np.multiply(y, np.dot(X, B))-np.log(1 + np.exp(np.dot(X, B))))


def log_likelihood_gradient(X, y, B, lmbda):
    return -np.dot(X.T, (y-sigmoid(np.dot(X, B))))


# NOT REQUIRED but to try to implement for fun
def L1_log_likelihood(X, y, B, lmbda):
    pass


# NOT REQUIRED but to try to implement for fun
def L1_log_likelihood_gradient(X, y, B, lmbda):
    """
    Must compute \beta_0 differently from \beta_i for i=1..p.
    \beta_0 is just the usual log-likelihood gradient
    # See https://aimotion.blogspot.com/2011/11/machine-learning-with-python-logistic.html
    # See https://stackoverflow.com/questions/38853370/matlab-regularized-logistic-regression-how-to-compute-gradient
    """
    pass


def minimize(X, y, loss, loss_gradient,
             eta=0.00001, lmbda=0.0,
             max_iter=1000, addB0=True,
             precision=0.00000001):
    # Check data dimension
    if X.ndim != 2:
        raise ValueError("X must be n x p for p features")
    n, p = X.shape
    if y.shape != (n, 1):
        raise ValueError(f"y must be n={n} x 1 not {y.shape}")

    if addB0:
        # add column of 1s to X
        B0_array = np.ones(shape=(n, 1))
        X = np.hstack([B0_array, X])
        p += 1

    # Initialize B and h and other parameters
    B = np.random.random_sample(size=(p, 1)) * 2 - 1  # make between [-1,1)
    h = np.zeros(shape=(p, 1))
    cost = 9e99
    eps = 1e-5  # prevent division by 0

    for step in range(0, max_iter):
        prev_B = B
        prev_cost = loss(X, y, prev_B, lmbda)

        lg_value = loss_gradient(X, y, B, lmbda)
        h += np.multiply(lg_value, lg_value)

        B = B - eta * lg_value / (np.sqrt(h) + eps)
        present_cost = loss(X, y, B, lmbda)
        if abs(prev_cost - present_cost) < precision:
            return B
    return B




class LinearRegression621:
    def __init__(self,
                 eta=0.00001, lmbda=0.0,
                 max_iter=1000):
        self.eta = eta
        self.lmbda = lmbda
        self.max_iter = max_iter

    def predict(self, X):
        n = X.shape[0]
        B0 = np.ones(shape=(n, 1))
        X = np.hstack([B0, X])
        return np.dot(X, self.B)

    def fit(self, X, y):
        self.B = minimize(X, y,
                           MSE,
                           loss_gradient,
                           self.eta,
                           self.lmbda,
                           self.max_iter)

class RidgeRegression621:
    def __init__(self,
                 eta=0.00001, lmbda=0.0,
                 max_iter=1000):
        self.eta = eta
        self.lmbda = lmbda
        self.max_iter = max_iter

    def predict(self, X):
        n = X.shape[0]
        B0 = np.ones(shape=(n, 1))
        X = np.hstack([B0, X])
        return np.dot(X, self.B)

    def fit(self, X, y):
        self.B = minimize(X, y,
                          loss_ridge,
                          loss_gradient_ridge,
                          self.eta,
                          self.lmbda,
                          self.max_iter)

class LogisticRegression621:
    def __init__(self,
                 eta=0.00001,lmbda=0.0,
                 max_iter=1000):
        self.eta = eta
        self.lmbda = lmbda
        self.max_iter = max_iter

    def predict(self,X):
        n = X.shape[0]
        B0 = np.ones(shape=(n, 1))
        X = np.hstack([B0, X])
        sigmoid_result = sigmoid(np.dot(X, self.B))
        return np.where(sigmoid_result > 0.5, 1, 0)

    def fit(self, X, y):
        self.B = minimize(X, y,
                          log_likelihood,
                          log_likelihood_gradient,
                          self.eta,
                          self.lmbda,
                          self.max_iter)


# NOT REQUIRED but to try to implement for fun
class LassoLogistic621:
    pass

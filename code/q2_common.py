"""
Shared utilities for Assignment 2, Question 2 (linear regression).
Data: A2Q2Data_train.csv (10000 x 101), A2Q2Data_test.csv (500 x 101).
First 100 columns = features, last column = target y.
"""
import numpy as np

TRAIN_PATH = "/home/claude/assignment2/data/A2Q2Data_train.csv"
TEST_PATH = "/home/claude/assignment2/data/A2Q2Data_test.csv"


def load_xy(path):
    data = np.loadtxt(path, delimiter=",")
    X = data[:, :-1]
    y = data[:, -1]
    return X, y


def analytical_ols(X, y):
    """
    w_ML = (X^T X)^{-1} X^T y
    Solves via np.linalg.solve for numerical stability (avoids explicitly
    inverting X^T X), which is mathematically equivalent to the closed-form
    normal-equations solution.
    """
    A = X.T @ X
    b = X.T @ y
    w_ml = np.linalg.solve(A, b)
    return w_ml


def mse(X, y, w):
    resid = X @ w - y
    return np.mean(resid ** 2)


def sse(X, y, w):
    resid = X @ w - y
    return np.sum(resid ** 2)

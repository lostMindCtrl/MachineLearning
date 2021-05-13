import numpy as np

class weightedLinearRegression:

    def __init__():
        self.data = []

def weightedLinearRegression():

    def prepareRegression(X):
        row, col      = X.shape
        ones          = np.ones((row, 1))
        new_X         = np.zeros((row, col+1))
        new_X[:, 0]   = ones[:, 0]
        new_X[:, 1:]  = X[:, :]
        return new_X

    def hypothesis(theta, x):
        return np.matmul(np.transpose(theta), np.transpose(x))

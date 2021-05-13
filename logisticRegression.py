import numpy  as np
import pandas as pd

def predict(theta, X):
    p                = sigmoid(np.matmul(X, theta))
    p[p >= 0.5]      = 1
    p[p < 0.5]       = 0
    return p

def hypothesis(theta, X):
    return sigmoid(np.transpose(theta)*X)

def sigmoid(z):
    sig = 1/(1 + np.exp(-z))
    return sig

def logisticRegression(iter, alpha, X, y):

    def prepareRegression(X):
        row, col      = X.shape
        ones          = np.ones((row, 1))
        new_X         = np.zeros((row, col+1))
        new_X[:, 0]   = ones[:, 0]
        new_X[:, 1:]  = X[:, :]
        return new_X

    def gradientDescent(alpha, m, X, y, theta):
        h     = sigmoid(np.matmul(X, theta))
        theta = theta - (alpha/m)*np.matmul(np.transpose(X), (h - y))
        grad  = (1/m) * np.matmul(np.transpose(X), (h - y))
        return theta, grad

    def costFunction(m, theta, X, y):
        h       = sigmoid(np.matmul(X, theta))
        J_theta = (1/m) * (np.matmul(np.transpose(-y), (np.log(h))) - np.matmul(np.transpose(1 - y), np.log(1 - h)))
        return J_theta

    def decisionBoundary():
        return 0

    def runRegression(iter, alpha, X, y):
        row, col = X.shape
        #print("Test cost:")
        #test_theta = np.transpose(np.matrix([-24, 0.2, 0.2]))
        #print(test_theta)
        #print("")
        #print(costFunction(row, test_theta, X, y))
        #print("\n\n")
        #theta, grad  = gradientDescent(alpha, row, X, y, test_theta)
        #print("<---------->")
        #print(grad)
        #print("<---------->")
        #print(X.shape)
        #print(y.shape)
        theta    = np.zeros((col, 1))
        cost     = np.zeros((iter, 1))
        for i in range(iter):
            theta, grad  = gradientDescent(alpha, row, X, y, theta)
            cost_theta   = costFunction(row, theta, X, y)
            cost[i, 0]   = cost_theta
        return theta, cost

    X           = prepareRegression(X)
    theta, cost = runRegression(iter, alpha, X, y)
    return theta, cost, X

def regularized_logisticRegression(iter, alpha, X, y, lam):

        def prepareRegression(X):
            row, col      = X.shape
            ones          = np.ones((row, 1))
            new_X         = np.zeros((row, col+1))
            new_X[:, 0]   = ones[:, 0]
            new_X[:, 1:]  = X[:, :]
            return new_X

        def costFunction(m, theta, X, y, lam, L):
            h = sigmoid(np.matmul(X, theta))
            J = (1/m)*(np.matmul(np.transpose(-y), (np.log(h))) - np.matmul(np.transpose(1 - y), np.log(1 - h))) + (lam/(2*m))*np.sum(np.power(np.matmul(L, theta), 2))
            return J

        def gradientDescent(m, alpha, theta, X, y, lam, L):
            h     = sigmoid(np.matmul(X, theta))
            theta = theta - (alpha/m)*np.matmul(np.transpose(X), (h - y)) + (lam/m)*np.matmul(L, theta)
            return theta

        def runRegression(iter, alpha, X, y, lam):
            row, col = X.shape
            theta    = np.zeros((col, 1))
            L        = np.identity(col)
            L[0, 0]  = 0
            cost     = np.zeros((iter, 1))
            for i in range(iter):
                theta      = gradientDescent(row, alpha, theta, X, y, lam, L)
                cost_theta = costFunction(row, theta, X, y, lam, L)
                cost[i, 0] = cost_theta
            return theta, cost
        X           = prepareRegression(X)
        theta, cost = runRegression(iter, alpha, X, y, lam)
        return theta, X, cost

import numpy as np

def hypothesis(theta, x):
    return np.matmul(np.transpose(theta), np.transpose(x))

def batch_regressionModel(iter, set, y, alpha):

    def prepareRegression(X):
        row, col      = X.shape
        ones          = np.ones((row, 1))
        new_X         = np.zeros((row, col+1))
        new_X[:, 0]   = ones[:, 0]
        new_X[:, 1:]  = X[:, :]
        return new_X

    def gradientDescent(theta, X, y, alpha, m):
        #delta = np.zeros(theta.shape)
        #delta[:,0] = (1/m)*np.sum(y - hypothesis(theta, X), axis=0).dot(X)
        #theta = theta + alpha*delta
        theta = theta + (alpha/m)*np.matmul(np.transpose(X), np.transpose(y - hypothesis(theta, X)))
        return theta

    def costFunction(theta, x, y, m):
        cost = (1/(2*m))*np.sum(np.square((hypothesis(theta, x) - y)))
        return cost

    def runRegression(iter, X, y, alpha):
        row, col = X.shape
        theta    = np.zeros((col,1))
        cost     = []
        y        = np.transpose(y)
        for i in range(iter):
            theta = gradientDescent(theta, X, y, alpha, row)
            cost.append(costFunction(theta, X, y, row))
        return theta, cost

    X           = prepareRegression(set)
    theta, cost = runRegression(iter, X, y, alpha)
    return theta, X, cost

def stochastic_regressionModel(iter, set, y, alpha):

    def prepareRegression(X):
        row, col      = X.shape
        ones          = np.ones((row, 1))
        new_X         = np.zeros((row, col+1))
        new_X[:, 0]   = ones[:, 0]
        new_X[:, 1:]  = X[:, :]
        return new_X

    def hypothesis(theta, x):
        return np.matmul(np.transpose(theta), x)

    def runRegression(iter, X, y, alpha):
        return theta, cost


def normalEquations_regressionModel(set, y):

    def prepareRegression(X):
        row, col      = X.shape
        ones          = np.ones((row, 1))
        new_X         = np.zeros((row, col+1))
        new_X[:, 0]   = ones[:, 0]
        new_X[:, 1:]  = X[:, :]
        return new_X

    def costFunction(theta, X, y):
        y_pred = np.matmul(X, theta)
        j = (1/2)*np.transpose((y_pred - y)) @ (y_pred - y)
        return j

    def normalEquation(X, y, theta):
        first   = np.linalg.inv(np.matmul(np.transpose(X), X))
        second  = (first @  np.transpose(X))
        theta   = np.matmul(second, y)
        return theta

    def runRegression(X, y):
        row, col = X.shape
        cost     = []
        theta    = np.ones((col, 1))
        theta    = normalEquation(X, y, theta)
        cost.append(costFunction(theta, X, y))
        return theta, cost

    X           = prepareRegression(set)
    theta, cost = runRegression(X, y)
    return theta, cost

def regularized_linearRegression(iter, alpha, lam, X, y):

    def prepareRegression(X):
        row, col      = X.shape
        ones          = np.ones((row, 1))
        new_X         = np.zeros((row, col+1))
        new_X[:, 0]   = ones[:, 0]
        new_X[:, 1:]  = X[:, :]
        return new_X

    def costFunction(m, X, y, theta, lam):
        h = np.matmul(X, theta)
        J = (1/2*m)*np.power(np.matmul(np.transpose(X), (h - y)), 2) + lam*np.sum(np.power(theta, 2))

    def gradientDescent(alpha, X, y, lam, m, theta):
        theta = theta*(1 - (lam/m)) - (alpha/m)*np.matmul(np.transpose(X), np.matmul(X, theta) - y)
        return theta

    def runRegression(iter, alpha, X, y, lam):
        row, col = X.shape
        theta    = np.zeros((col, 1))
        cost     = np.zeros((iter, 1))
        for i in range(iter):
            theta      = gradientDescent(alpha, X, y, lam, row, theta)
            cost[i, 0] = costFunction(row, X, y, theta, lam)
        return theta, cost
    X           = prepareRegression(X)
    theta, cost = runRegression(iter, alpha, X, y, lam)
    return theta, X, cost

def regularized_normalEquation(X, y, lam):

    def prepareRegression(X):
        row, col      = X.shape
        ones          = np.ones((row, 1))
        new_X         = np.zeros((row, col+1))
        new_X[:, 0]   = ones[:, 0]
        new_X[:, 1:]  = X[:, :]
        return new_X

    def normalEquation(X, y, L, lam, theta):
        theta = np.matmul(np.linalg.inv(np.matmul(np.transpose(X), X) + (lam*L)), np.transpose(X) @ y)
        return theta

    def runRegression(X, y, lam):
        row, col = X.shape
        L        = np.identity(col)
        L[0,0]   = 0
        theta    = np.zeros((col, 1))
        theta    = normalEquation(X, y, L, lam, theta)
        return theta

    X     = prepareRegression(X)
    theta = runRegression(X, y, lam)
    return theta, X

#//A different representation of batch_gradientDescent//
#delta[:,0] = (1/m)*np.sum(hypothesis(theta, X) - y, axis=0).dot(X)
#theta = theta - alpha*delta

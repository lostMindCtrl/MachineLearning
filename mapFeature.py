import numpy as np

def featureMapping(X1, X2, deg):
    row      = X1.shape[0]
    output   = np.ones((row, 28))

    for x in range(28):
        for i in range(1, deg):
            for j in range(i):
                output[:, x] = np.power(X1, (i-j)) * np.power(X2, j)
    return output

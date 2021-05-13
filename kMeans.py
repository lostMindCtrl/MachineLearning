import numpy as np
import numpy.matlib

def kMeans(X, iter, K):
    row, col     = X.shape
    clusters     = np.zeros((K, col))
    init_random  = np.random.permutation(row)
    clusters     = X[init_random, :]
    pre_clusters = clusters
    result       = np.zeros((row, 1))

    for i in range(iter):
        for j in range(row):
            dist           = np.zeros((1, K))
            for y in range(K):
                dist[0, y]  = np.sqrt(np.sum(np.power(X[j, :] - clusters[y, :], 2)))
            min            = np.argmin(dist)
            result[j, 0]   = min

    pre_clusters = clusters

    for j in range(K):
        cluster_count = np.count_nonzero(result == j)
        cc_matrix     = np.matlib.repmat(cluster_count, 1, col)
        new_clusters  = X.dot(np.transpose(cc_matrix))
        clusters[j, :] = np.sum(new_clusters)/(cluster_count)

    return result

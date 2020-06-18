import numpy as np
from PIL import  Image
import matplotlib.pyplot as plt
import sys


def dist(x, y):
    (rowx, colx) = x.shape
    (rowy, coly) = y.shape
    xy = np.dot(x, y.T)
    x2 = np.repeat(np.reshape(np.sum(np.multiply(x, x), axis=1), (rowx, 1)), repeats=rowy, axis=1)
    y2 = np.repeat(np.reshape(np.sum(np.multiply(y, y), axis=1), (rowy, 1)), repeats=rowx, axis=1).T
    dis = x2 + y2 - 2 * xy
    return dis

def kmeans(X:np.ndarray, k:int, centroids=None, tolerance=1e-2):
    X = X.astype('int')
    length = X.shape[0]
    feature_num = X.shape[1]
    if centroids == None:
        centroid_idx = list(np.random.choice(length, size=k, replace=False))
        centroids = X[centroid_idx]

    elif centroids == 'kmeans++':
        first_idx = np.random.choice(length, size=1, replace=False)
        index = []
        index.append(int(first_idx))
        for i in range(k - 1):
            EucDist = dist(X, X[index])
            index.append(np.argmax(np.min(EucDist, axis=1)))
        centroids = X[index]
    EucDist = dist(X, centroids) # [n, k]
    idx = np.argmin(EucDist, axis=1)
    old_idx = idx
    while True:
        for i in range(k):
            points = X[idx == i]
            centroids[i] = points.mean(axis=0)
        EucDist = dist(X, centroids)
        idx = np.argmin(EucDist, axis=1)
        if np.sum(np.abs(X[idx] - X[old_idx])**2)**(1./2) < tolerance:
            # clusters = []
            # for i in range(k):
            #     clusters.append(X[idx == i])
            # return centroids, clusters
            return centroids, idx
        else:
            old_idx = idx

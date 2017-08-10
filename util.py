import numpy as np
from scipy.io import loadmat


def y2indicator(y):
    N = len(y)
    ind = np.zeros((N, 10))
    for i in range(N):
        ind[i, y[i]] = 1
    return ind


def error_rate(p, t):
    return np.mean(p != t)


def flatten(X):
    # input will be (32, 32, 3, N)
    # output will be (N, 3072)
    N = X.shape[-1]
    flat = np.zeros((N, 3072))
    for i in range(N):
        flat[i] = X[:, :, :, i].reshape(3072)
    return flat


def get_data():
    train = loadmat('train_32x32.mat')
    test = loadmat('test_32x32.mat')
    return train, test

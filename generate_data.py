'''
generates synthetic data with the following constraints:
1. X ~ Unif(-1, 1)
2. betas ~ Normal(0, sqrt(N))
3. Y = 0 if X*beta <= 0, Y = 1 if X*beta > 0 for arbitrary beta
'''

import numpy as np

def generate_x(N, K):
    X = np.empty(shape=(N, K))
    for i in range(N):
        X[i] = np.random.uniform(low=-1, high=1, size=K)
    return X

def generate(N, K, beta):
    X = generate_x(N, K)
    y = np.empty(shape=N)
    for i in range(N):
        rho = np.random.standard_normal()
        xbeta = np.dot(X[i], beta)
        if xbeta + rho > 0:
            y[i] = 1
        else:
            y[i] = 0
    return X, y

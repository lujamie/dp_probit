'''
baseline mechanisms for differential privacy
'''

import numpy as np
import math
import random


def noise_param(eps, delta, sensitivity, d, var):
    if var == "x":
        return (d*(sensitivity**2)/eps**2)*(2.0*math.log(1.25/delta)) # x_sigma2
    elif var == "y":
        #y_sigma2 = (1/eps**2)*(2.0*math.log(1.25/delta))    # sensitivity of y's = 1
        return math.exp(eps / 2) / (math.exp(eps / 2) + 1) # py
    else:
        print("invalid variable")
        return 

def rr(p, yi):
    '''
    wp e^eps / 1+ e^eps, output true answer, ow flip
    '''
    if (random.uniform(0, 1) < p):
        return yi
    else:
        return 1 - yi


def privatize_gauss(X, y, beta, x_sigma2, y_sigma2):
    '''
    add Gaussian noise to X's and y's
    '''
    n = np.shape(y)[0]
    d = np.shape(X)[1]
    
    X_priv = np.copy(X)
    y_priv = np.zeros(n)

    for i in range(n):
        x_noise = np.random.normal(loc=0, scale=math.sqrt(x_sigma2), size=d)
        X_priv[i] += x_noise

        y_noise = np.random.normal(loc=0, scale=math.sqrt(y_sigma2))
        if y_noise + float(y[i]) >= 0.5:
            y_priv[i] += 1

    return X_priv, y_priv


# def privatize(X, y, sigma2, eps):
#     '''
#     Gaussin noise for x's, randomized response for y's
#     eps = overall budget
#     sigma2 = scale of Gaussian noise for X's 
#     '''
#     n = np.shape(y)[0]
#     d = np.shape(X)[1]
    
#     X_priv = np.copy(X)
#     y_priv = np.zeros(n)

#     for i in range(n):
#         x_noise = np.random.normal(loc=0, scale=math.sqrt(sigma2), size=d)
#         X_priv[i] += x_noise
#         p = math.exp(eps) / (math.exp(eps) + 1)
#         y_priv[i] = rr(p, y[i])

#     return X_priv, y_priv

def privatize(X, y, sigma2, py):
    '''
    Gaussian noise for x's, randomized response for y's
    py = probabiliy of outputting true y
    sigma2 = scale of Gaussian noise for X's
    '''
    n = np.shape(y)[0]
    d = np.shape(X)[1]

    X_priv = np.copy(X)
    y_priv = np.zeros(n)

    for i in range(n):
        x_noise = np.random.normal(loc=0, scale=math.sqrt(sigma2), size=d)
        X_priv[i] += x_noise

        y_priv[i] = rr(py, y[i])

    return X_priv, y_priv

# return sampling size
def sample_sz(n, eps, delta):
    m = 1
    km = 0
    for k in range(n**2):
        fac = m*((math.exp(eps) - 1) / n)
        #k = eps * math.log(1 + fac)
        delta_p = delta - (k*delta/(n*math.log(n)))
        if delta_p < 0:
            break
        if math.log(1 + fac) * math.sqrt(2*k*math.log(1/delta_p)) + k*math.log(1 + fac)*fac > eps:
            continue
        if k * m > km:
            km = k*m
    print(f"sampling size = {km}")
    return km

# privatization for subsampling method
def sampling_params(n, d, eps, delta, sensitivity):
    half_delta = delta / 2
    half_eps = eps / 2
    y_sigma2 = n**2 * math.log(1/half_delta) * math.log(1/half_delta) / half_eps**2
    x_sigma2 = d * sensitivity**2 * y_sigma2
    return y_sigma2, x_sigma2

def sampling_privatize(X, y, beta, y_sigma2, x_sigma2, sz):
    n = np.shape(X)[0]
    d = np.shape(X)[1]
    sample = np.random.randint(0, n-1, size=sz)
    X_sample = np.empty((sz, d))
    y_sample = np.empty(sz)
    for i in range(sz):
        indx = sample[i]
        X_sample[i] = X[indx]
        y_sample[i] = y[indx] 
    X_priv, y_priv = privatize(X_sample, y_sample, beta, x_sigma2, y_sigma2)
    return X_priv, y_priv
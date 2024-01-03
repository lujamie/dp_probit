from scipy.optimize import minimize
from scipy.stats import norm
import math
import numpy as np
from generate_data import generate
import random
from probit import predict
from probit import run_probit


def nll(param, x, y, py):
    '''
    returns the negative log-likelihood function
    x, y: privatized data
    param: gamma * beta
    py: probably of outputting true y in RR
    '''
    n = np.shape(x)[0]
    ll = 0
    for i in range(n):
        mu = norm.cdf(np.dot(x[i], param))
        ll += y[i] * math.log((py * mu) + (1-py)*(1-mu)) + (1-y[i]) * math.log((py * (1-mu)) + (1-py)*mu)
    return -1 * ll
    
def nll_wrapper(x, y, py):
    return (lambda b: nll(b, x, y, py))

def constraint(xi, sigma2_x):
    print(np.linalg.norm([x / np.sqrt(1 + sigma2_x) for x in xi]))
    return 1 - np.linalg.norm([x / np.sqrt(1 + sigma2_x) for x in xi])

def con_optimize(x, y, py, sigma2_x):
    # tol = 1e-6
    # cons = ({'type': 'eq', 'fun': lambda x:  constraint(x, sigma2_x) + tol}
    # )
    d = np.shape(x)[1]
    start = [1.] * d
    res = minimize(nll_wrapper(x, y, py), start, method="SLSQP", tol=1e-8)
    print(res.x, res.message)
    return res.x

def get_beta(xi, sigma2):
    xi_norm = np.linalg.norm(xi)
    u = [x/xi_norm for x in xi]
    xi_norm2 = xi_norm**2
    b = np.sqrt(abs(xi_norm2 / (1 - xi_norm2*sigma2)))
    return [x * b for x in u]


def rr(py, yi):
    '''
    with probability py, output yi, otherwise flip
    '''
    if (random.uniform(0, 1) < py):
        return yi
    else:
        return 1 - yi

def priv_rr(X, y, sigma2, py):
    '''
    gaussian noise for x's, randomized response for y's
    sigma2 = scale of gaussian noise
    py = probability that we output true y
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

# n = 400
# d = 5
# sigma2 = 1.0
# beta = np.random.normal(loc=0, scale=math.sqrt(math.sqrt(n)), size=d)
# print(beta)
# X, y = generate(n, d, beta)
# X_test, y_test = generate(n, d, beta)

# py = math.exp(2) / (1 + math.exp(2)) # fixing epsilon_y = 2
# X_priv, y_priv = priv_rr(X, y, sigma2, py)
# xi = con_optimize(X_priv, y_priv, py, sigma2)
# betah = get_beta(xi, sigma2)
# print("beta", betah)
# score = predict(betah, X_test, y_test)

# print("score:", score)

def fixed_noise(X, y, X_test, y_test, sigma2, py):
    X_priv, y_priv = priv_rr(X, y, sigma2, py)
    xi = con_optimize(X_priv, y_priv, py, sigma2)
    betah = get_beta(xi, sigma2)
    score = predict(betah, X_test, y_test)
    print(f"n = {np.shape(y)[0]}, d = {np.shape(X)[1]}, sigma2 = {sigma2}, py = {py}, score = {score}")
    return score

def run_fixed_noise():
    # Define parameters
    N = [4000]
    D = [5]
    sigma2 = np.arange(start=0, stop=2.0, step=0.2)[::-1] # reversed [1, 0.9,..., 0.1, 0]
    py = math.exp(2) / (1 + math.exp(2)) # fixing epsilon_y = 2
    num_trials = 8

    max_noise = {}
    results = {}

    for n in N:
        for d in D:
            beta = np.random.normal(loc=0, scale=math.sqrt(math.sqrt(n)), size=d)
            X, y = generate(n, d, beta)
            X_test, y_test = generate(n, d, beta)
            
            scores = []
            # sigma2 = 1.8
            max_sigma2 = 0
            reached = False
            for s in sigma2:
                score = 0
                for i in range(num_trials):
                    score += fixed_noise(X, y, X_test, y_test, s, py)
                score /= num_trials
                if score > 0.8 and not reached: # benchmark score
                    max_sigma2 = s
                    scores.append(score)
                    reached = True
                #sigma2 -= 0.2
                scores.append(score)
            print(f"(n, d) = ({np.shape(y)[0]}, {d}), scores = {scores}")
            max_noise[(n, d)] = max_sigma2
            results[(n, d)] = scores

    print(f"scores: {results}")
    print(f"min sigmas: {max_noise}")
'''
Experiments for private and nonprivate probit regression
error metric is the log likelihood of estimators
tries different values of the privacy parameter
'''

from generate_data import generate
from probit import predict, run_probit
from mechanisms import noise_params, privatize_rr
from priv_opt import optimize_rr, optimize_xi
from constrained_opt import get_beta
import numpy as np
import math
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
import pandas as pd
import random

# initialize parameters
d = 5   # dimensions
#N = 600  # sample size
num_trials = 2

# privacy parameters
eps = np.arange(start=260, stop=300, step=10)    # overall epsilon values after shuffling
num_eps = eps.shape[-1]
sensitivity = 2.0 # range that X values take


# regular Probit regression
def run_nondp(X, y, X_test, y_test):
    nondp_res = []
    m = num_eps
    for i in range(m):
        params = run_probit(X, y)
        nondp_res.append(predict(params, X_test, y_test))
    return max(nondp_res)

def dp_noshuffle(X, y, X_test, y_test, d, prob:bool):
    dp_res = []
    n = np.shape(y)[0]
    delta = 1/n
    for e in eps:
        x_sigma2, y_sigma2 = noise_params(e, delta, sensitivity, d)
        X_priv, y_priv = privatize_rr(X, y, x_sigma2, e)
        score = 0
        for i in range(num_trials):
            if prob:
                params = run_probit(X_priv, y_priv)
            else:
                params = optimize_rr(X_priv, y_priv, math.sqrt(y_sigma2), x_sigma2)
            gamma = 1/math.sqrt(1+(x_sigma2*np.square(np.linalg.norm(params))))
            p = math.exp(e) / (math.exp(e) + 1)
            score += predict(params, X_test, y_test)
        score /= num_trials
        print(f"n = {np.shape(y)[0]}, d = {d}, eps = {e}, delta = {delta}, score = {score}")
        dp_res.append(score)
    return dp_res

def run_dp_noshuffle():
    n = 400
    beta = np.random.normal(loc=0, scale=math.sqrt(math.sqrt(n)), size=d)
    X, y = generate(n, d, beta)
    X_test, y_test = generate(n, d, beta)
    res = dp_noshuffle(X, y, X_test, y_test, d, prob=True)
    plt.plot(eps, res, label="DP")
    plt.show()
    

# def run_dp_x(X, y, X_test, y_test, beta, d):
#     dp_res = []
#     for e in eps:
#         x_sigma2, _ = noise_params(e, delta, sensitivity, d)
#         X_priv, _ = privatize(X, y, beta, x_sigma2, 0)
#         params = optimize_x(X_priv, y, x_sigma2)
#         score = predict(params, X_test, y_test)
#         print(f"eps = {e}, delta = {delta}, score = {score}")
#         dp_res.append(score)
#     return dp_res

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
    

def fixed_noise(X, y, X_test, y_test, sigma2, py):
    X_priv, y_priv = priv_rr(X, y, sigma2, py)
    xi = optimize_xi(X_priv, y_priv, py, sigma2)
    #gamma = 1/math.sqrt(1+(sigma2*np.square(np.linalg.norm(params))))
    betah = get_beta(xi, sigma2)
    score = predict(betah, X_test, y_test)
    print(f"n = {np.shape(y)[0]}, d = {d}, sigma2 = {sigma2}, py = {py}, score = {score}")
    return score
    
def run_fixed_noise():
    # Define parameters
    N = [40000]
    D = [5]
    sigma2 = np.arange(start=0, stop=1, step=0.1)[::-1] # reversed [1, 0.9,..., 0.1, 0]
    py = math.exp(2) / (1 + math.exp(2)) # fixing epsilon_y = 2
    num_trials = 8

    min_noise = {}
    results = {}

    for n in N:
        delta = 1/n
        
        for d in D:
            beta = np.random.normal(loc=0, scale=math.sqrt(math.sqrt(n)), size=d)
            X, y = generate(n, d, beta)
            X_test, y_test = generate(n, d, beta)
            
            scores = []
            sigma2 = 1.8
            min_sigma2 = 0
            while sigma2 >= 0: 
                score = 0
                for i in range(num_trials):
                    score += fixed_noise(X, y, X_test, y_test, sigma2, py)
                score /= num_trials
                if score > 0.8: # benchmark score
                    min_sigma2 = sigma2
                    scores.append(score)
                    break
                sigma2 -= 0.2
                scores.append(score)
            print(f"(n, d) = ({np.shape(y)[0]}, {d}), scores = {scores}")
            min_noise[(n, d)] = min_sigma2
            results[(n, d)] = scores

    print(f"scores: {results}")
    print(f"min sigmas: {min_noise}")
    df = pd.DataFrame([results, min_noise]).T
    df.columns = ['d{}'.format(i) for i, col in enumerate(df, 1)]



def check_grad(X, y, X_test, y_test, sigma2, py):
    X_priv, y_priv = priv_rr(X, y, sigma2, py)
    params, ll, grad = optimize_rr(X_priv, y_priv, py, sigma2)
    print(f"log-likelihood value is {ll}, gradient is {grad}")
    gamma = 1/math.sqrt(1+(sigma2*np.square(np.linalg.norm(params))))
    score = predict(params, X_test, y_test)
    print(f"score = {score}")
    return params, ll, grad, score

def run_grad_check():
    N = [400, 1000, 4000, 6000]
    d = 5
    sigma2 = np.arange(start=0, stop=1, step=0.1)[::-1] # reversed [1, 0.9,..., 0.1, 0]
    py = math.exp(2) / (1 + math.exp(2)) # fixing epsilon_y = 2
    grads = []
    for n in N:
        beta = np.random.normal(loc=0, scale=math.sqrt(math.sqrt(n)), size=d)
        X, y = generate(n, d, beta)
        X_test, y_test = generate(n, d, beta)
        delta = 1/n
        sigma2 = 2.0
        params, ll, res, score = check_grad(X, y, X_test, y_test, sigma2, py)
        _, grad = res
        grads.append([n, ll.numpy(), grad.numpy(), score, beta, params])
    df = pd.DataFrame(grads, columns=["n", "loss", "gradients", "score", "true beta", "estimated beta"])
    df.to_csv("grad.csv")


run_fixed_noise()
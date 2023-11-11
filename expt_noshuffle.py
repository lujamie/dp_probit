'''
Experiments for private and nonprivate probit regression
error metric is the log likelihood of estimators
tries different values of the privacy parameter
'''

from generate_data import generate
from probit import predict, run_probit
from mechanisms import noise_params, privatize, privatize_rr
from priv_opt import optimize_rr, optimize_x
import numpy as np
import math
import matplotlib.pyplot as plt
import random

# initialize parameters
d = 5   # dimensions
#N = 600  # sample size
num_trials = 2

# privacy parameters
eps = np.arange(start=1, stop=52, step=10)    # overall epsilon values after shuffling
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

def run_dp_noshuffle(X, y, X_test, y_test, beta, d, prob:bool):
    dp_res = []
    for e in eps:
        x_sigma2, y_sigma2 = noise_params(e, delta, sensitivity, d)
        X_priv, y_priv = privatize_rr(X, y, x_sigma2, e)
        if prob:
            params = run_probit(X_priv, y_priv)
        else:
            params = optimize_rr(X_priv, y_priv, math.sqrt(y_sigma2), x_sigma2)
        score = predict(params, X_test, y_test)
        print(f"n = {np.shape(y)[0]}, d = {d}, eps = {e}, delta = {delta}, score = {score}, probit = {prob}")
        dp_res.append(score)
    return dp_res

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
    gaussin noise for x's, randomized response for y's
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
    

def run_fixed_noise(X, y, X_test, y_test, sigma2, py):
    X_priv, y_priv = priv_rr(X, y, sigma2, py)
    params = optimize_rr(X_priv, y_priv, py, sigma2)
    score = predict(params, X_test, y_test)
    print(f"n = {np.shape(y)[0]}, d = {d}, sigma2 = {sigma2}, py = {py}, score = {score}")
    return score
    
# Define parameters
N = np.arange(start=10000, stop=60000, step=20000)
D = [5, 10, 25]
sigma2 = np.arange(start=0, stop=1, step=0.1)[::-1] # reversed [1, 0.9,..., 0.1, 0]
py = math.exp(2) / (1 + math.exp(2)) # fixing epsilon_y = 2
num_trials = 3


min_noise = {}
results = {}
fig = plt.figure()
idx = 1

for n in N:
    delta = 1/n
    for d in D:
        beta = np.random.normal(loc=0, scale=math.sqrt(math.sqrt(n)), size=d)
        X, y = generate(n, d, beta)
        X_test, y_test = generate(n, d, beta)
        
        scores = []
        sigma2 = 2.8
        min_sigma2 = 0
        while sigma2 >= 0: 
            score = 0
            for i in range(num_trials):
                score += run_fixed_noise(X, y, X_test, y_test, sigma2, py)
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
        
        ax = fig.add_subplot(3,3,idx)
        idx += 1
        ax.title(f"n = {N}, d = {d}, min_sigma2 = {min_sigma2}")
        ax.plot(np.arange(min_sigma2, 2.8, 0.2), scores)
        ax.xlabel('Gaussian noise σ^2')
        ax.ylabel('Classification error')

plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.4, 
                    hspace=0.4)
plt.show()
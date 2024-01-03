'''
Experiments for private and nonprivate probit regression
error metric is the log likelihood of estimators
tries different values of the privacy parameter
'''

from generate_data import generate
from probit import predict, run_probit
from mechanisms import noise_param, privatize
from priv_opt import optimize_rr, optimize_scipy
import numpy as np
import math
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
import pandas as pd

# initialize parameters
d = 5   # dimensions
#N = 600  # sample size
num_trials = 2

# privacy parameters
eps = np.arange(start=0.1, stop=5, step=0.5)    # overall epsilon values after shuffling
num_eps = eps.shape[-1]
sensitivity = 2.0 # range that X values take


# regular Probit regression
def run_nondp(X, y, X_test, y_test):
    nondp_res = []
    m = 8 #num_trials
    for i in range(m):
        params = run_probit(X, y)
        nondp_res.append(predict(params, X_test, y_test))
    return max(nondp_res)

def dp_noshuffle(X, y, X_test, y_test, probit:bool):
    dp_res = []
    n = np.shape(y)[0]
    delta = 1/n
    for e in eps:
        x_sigma2 = 0
        py = noise_param(e, delta, sensitivity, d, var="y")
        X_priv, y_priv = privatize(X, y, x_sigma2, py)
        score = 0
        for i in range(num_trials):
            if probit:
                params = run_probit(X_priv, y_priv)
            else:
                params = optimize_rr(X_priv, y_priv, py, x_sigma2)
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
    prob_res = dp_noshuffle(X, y, X_test, y_test, probit=True)
    dp_res = dp_noshuffle(X, y, X_test, y_test, probit=False)
    print("beta:", beta)
    plt.plot(eps, prob_res, label="Probit")
    plt.plot(eps, dp_res, label="Gaussian")
    plt.xlabel("epsilon")
    plt.ylabel("classification rate")
    plt.title(f"n = {n}, d = {d}, delta = {1/n}")
    plt.legend()
    plt.show()
    

def fixed_noise(X_priv, y_priv, X_test, y_test, sigma2, py):
    param = optimize_scipy(X_priv, y_priv, py, sigma2)
    score = predict(param, X_test, y_test)
    print(f"n = {np.shape(y_priv)[0]}, d = {d}, sigma2 = {sigma2}, py = {py}, score = {score}")
    return score

def run_no_noise():
    n = 400
    d = 1
    num_trials = 8
    sigma2 = 1
    py = 1
    #beta = np.random.normal(loc=0, scale=math.sqrt(math.sqrt(n)), size=d)
    beta = 5.0
    X, y = generate(n, d, beta)
    X_priv, y_priv = privatize_rr_fixed(X, y, sigma2, py)
    if np.array_equal(X, X_priv) and np.array_equal(y, y_priv):
        print("no noise added")
    else:
        print("no noise failed")
        #return
    X_test, y_test = generate(n, d, beta)
    nondp_score = run_nondp(X_priv, y_priv, X_test, y_test)
    score = 0
    for i in range(num_trials):
        score += fixed_noise(X_priv, y_priv, X_test, y_test, sigma2, py)
    score /= num_trials
    print(f"dp score = {score}, nondp score = {nondp_score}")
    
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


run_dp_noshuffle()
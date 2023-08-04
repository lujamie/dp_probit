'''
Experiments for private and nonprivate probit regression
error metric is the log likelihood of estimators
tries different values of the privacy parameter
'''

from generate_data import generate
from mechanisms import probit_reg, gauss_mech
import numpy as np
import math
import matplotlib.pyplot as plt

# initialize parameters
K = 2   # dimensions
N = 100000  # sample size
num_trials = 3
error = "BETA-NORM"  # error metric: LOG-LIKE, BETA-NORM, PRED

# privacy parameters
eps = np.arange(start=1, stop=50, step=5)
num_eps = eps.shape[-1]
delta = 0.1
sensitivity = 2.0 # range that X values take
beta = np.random.normal(loc=0, scale=math.sqrt(math.sqrt(N)), size=K)

def run_dp(X, y):
    dp_res = np.zeros(shape=num_eps)
    for i in range(num_eps):
        X_priv = gauss_mech(X, N, K, eps[i], delta, sensitivity)
        for j in range(num_trials):
            dp_res[i] += probit_reg(X_priv, y, error, N, K, beta)
        dp_res[i] /= num_trials
    return dp_res

def run_nondp(X, y):
    nondp_res = 0
    for i in range(num_trials):
        nondp_res += probit_reg(X, y, error, N, K, beta)
    nondp_res /= num_trials
    return nondp_res

def run_trials():
    X, y = generate(N, K, beta)
    
    nondp_res = run_nondp(X, y)
    dp_res = run_dp(X, y)

    print("NONDP RESULT: ", nondp_res)
    print("DP RESULT: ",dp_res)

    plt.plot(eps, dp_res, label='DP')
    plt.plot(eps, (nondp_res * np.ones(shape=num_eps)), label='non-DP')
    plt.xlabel('espilon')
    plt.ylabel('||beta - beta_hat||')
    plt.legend()
    plt.show()

run_trials()
print("BETA:", beta)
'''
Experiments for private and nonprivate probit regression
error metric is the log likelihood of estimators
tries different values of the privacy parameter
'''

from cmath import sqrt
from generate_data import generate
from mechanisms import nondp_loglike, probit_err, privatize, run_probit
import numpy as np
import math
import matplotlib.pyplot as plt

# initialize parameters
K = 5   # dimensions
N = 100000  # sample size
num_trials = 1
error = "PRED"  # error metric: LOG-LIKE, BETA-NORM, PRED, LOG-LIKE2

# privacy parameters
eps = np.arange(start=0.1, stop=5, step=0.1)
num_eps = eps.shape[-1]
delta = 0.1
sensitivity = 2.0 # range that X values take
beta = np.random.normal(loc=0, scale=math.sqrt(math.sqrt(N)), size=K)

def run_dp(X, y, X_test, y_test):
    dp_res = np.zeros(shape=num_eps)
    for i in range(num_eps):
        X_priv = privatize(X, N, K, eps[i], delta, sensitivity, beta)
        for j in range(num_trials):
            if error == "LOG-LIKE2":
                dp_res[i] += nondp_loglike(X, y, X_priv)
            else:
                dp_res[i] += probit_err(X_priv, y, error, N, K, beta, X_test, y_test)
        dp_res[i] /= num_trials
    return dp_res

def run_nondp(X, y, X_test, y_test):
    nondp_res = []
    m = num_trials * num_eps
    for i in range(m):
        if error == "LOG-LIKE2":
            nondp_res += probit_err(X, y, "LOG-LIKE", N, K, beta, X_test, y_test)
        else:
            nondp_res.append(probit_err(X, y, error, N, K, beta, X_test, y_test))
    #nondp_res /= m
    return max(nondp_res)

def run_trials():
    X, y = generate(N, K, beta)
    X_test, y_test = generate(N, K, beta)
    nondp_res = run_nondp(X, y, X_test, y_test)
    dp_res = run_dp(X, y, X_test, y_test)

    print("NONDP RESULT: ", nondp_res)
    print("DP RESULT: ",dp_res)

    plt.style.use('dark_background')
    plt.plot(eps, dp_res, label='DP')
    plt.plot(eps, (nondp_res * np.ones(shape=num_eps)), label='non-DP')
    plt.xlabel('espilon')
    plt.ylabel('classification error')
    plt.legend()
    plt.show()

run_trials()
print("BETA:", beta)
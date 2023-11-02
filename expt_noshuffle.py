'''
Experiments for private and nonprivate probit regression
error metric is the log likelihood of estimators
tries different values of the privacy parameter
'''

from generate_data import generate
from probit import predict, run_probit
from mechanisms import noise_params, privatize, privatize_rr
from priv_opt import optimize, optimize_x
import numpy as np
import math
import matplotlib.pyplot as plt

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

def run_dp_noshuffle(X, y, X_test, y_test, beta, rr:bool):
    dp_res = []
    for e in eps:
        x_sigma2, y_sigma2 = noise_params(e, delta, sensitivity, d)
        if rr:
            X_priv, y_priv = privatize_rr(X, y, beta, x_sigma2, e)
        else:
            X_priv, y_priv = privatize(X, y, beta, x_sigma2, y_sigma2)
        params = optimize(X_priv, y_priv, math.sqrt(y_sigma2), x_sigma2)
        score = predict(params, X_test, y_test)
        print(f"eps = {e}, delta = {delta}, score = {score}, rr = {rr}")
        dp_res.append(score)
    return dp_res

def run_dp_x(X, y, X_test, y_test, beta):
    dp_res = []
    for e in eps:
        x_sigma2, _ = noise_params(e, delta, sensitivity, d)
        X_priv, _ = privatize(X, y, beta, x_sigma2, 0)
        params = optimize_x(X_priv, y, x_sigma2)
        score = predict(params, X_test, y_test)
        print(f"eps = {e}, delta = {delta}, score = {score}")
        dp_res.append(score)
    return dp_res




for N in [600]:
    delta = 1/N
    beta = np.random.normal(loc=0, scale=math.sqrt(math.sqrt(N)), size=d)
    nondp = []
    rr = []
    x_res = []
    for i in range(num_trials):
        X, y = generate(N, d, beta)
        X_test, y_test = generate(N, d, beta)
        nondp.append(run_nondp(X, y, X_test, y_test))
        xy = run_dp_noshuffle(X, y, X_test, y_test, beta, rr=True)
        x = run_dp_x(X, y, X_test, y_test, beta)
        x_res.append(x)
        rr.append(xy)

        print(f'''round {i}
            Non-DP: {max(nondp)},
            Privatize X, Y (RR): {xy},
            Privatize X results: {x}
        ''')

    nondp_mean = np.mean(nondp)

    rr_mean = np.mean(rr, axis=0)
    rr_err = np.std(rr, axis=0)
    x_mean = np.mean(x_res, axis=0)
    x_err = np.std(x_res, axis=0)

    plt.plot(eps, (nondp_mean * np.ones(shape=num_eps)), label="Non-DP")
    plt.errorbar(eps, x_mean, yerr=x_err, markersize=8, capsize=10, label="Privatize X")
    plt.errorbar(eps, rr_mean, yerr=rr_err, markersize=8, capsize=10, label="RR")

    plt.title(f"Privatie X, no shufle, n = {N}, d = {d}, delta = {delta}")
    plt.xlabel('espilon')
    plt.ylabel('classification error')
    plt.legend()
    plt.show()
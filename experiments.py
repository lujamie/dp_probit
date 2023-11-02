'''
Experiments for private and nonprivate probit regression
error metric is the log likelihood of estimators
tries different values of the privacy parameter
'''

from generate_data import generate, torch_data
from probit import predict, run_probit
from mechanisms import noise_params, privatize, privatize_rr
from localbudget import binary_search, compose
from priv_opt import optimize, optimize_rr
from sgd import dp_sgd
import numpy as np
import math
import matplotlib.pyplot as plt

# initialize parameters
d = 5   # dimensions
N = 10000  # sample size
num_trials = 2

# privacy parameters
eps = np.arange(start=0.5, stop=4, step=1)    # overall epsilon values after shuffling
num_eps = eps.shape[-1]
delta = 1/N
sensitivity = 2.0 # range that X values take
beta = np.random.normal(loc=0, scale=math.sqrt(math.sqrt(N)), size=d)


# regular Probit regression
def run_nondp(X, y, X_test, y_test):
    nondp_res = []
    m = num_eps
    for i in range(m):
        params = run_probit(X, y)
        nondp_res.append(predict(params, X_test, y_test))
    return max(nondp_res)

# shuffling + averaging
def averaging(X, y, eps_0, delta_0, k, beta, privll:bool):
    sum_params = np.zeros(shape=(d, 1))
    for i in range(k):
        x_sigma2, y_sigma2 = noise_params(eps_0, delta_0, sensitivity, d)
        X_priv, y_priv = privatize(X, y, beta, x_sigma2, y_sigma2, eps_0)
        if privll:
            params = optimize(X_priv, y_priv, math.sqrt(y_sigma2), x_sigma2).reshape(d, 1)
        else:
            params = run_probit(X_priv, y_priv)
        sum_params += params
    avg_params = sum_params / (np.ones((d, 1)) * k)
    print("AVG PARAMS: ", avg_params)
    return avg_params

# local Gaussion + shuffle
# def run_dp_shuffle(X, y, X_test, y_test, beta, avg:bool, sampling:bool, privll:bool):
#     dp_res = []
#     budget = []
#     for e in eps:
#         if avg:
#             eps_0, delta_0, k = local_budget_renyi(e, delta, N, avg=True)
#             avg_params = averaging(X, y, eps_0, delta_0, k, beta, privll)
#             score = predict(avg_params, X_test, y_test)
#             print(f"eps = {e}, score = {score}")
#         else:
#             if sampling:
#                 sz = sample_sz(N, e, delta)
#                 y_sigma2, x_sigma2 = sampling_params(sz, d, e, delta, sensitivity)
#                 X_priv, y_priv = sampling_privatize(X, y, beta, y_sigma2, x_sigma2, sz)
#                 #print(f"subsampling noise for y = {y_sigma2}")
#             else:
#                 eps_0, delta_0, k = local_budget_renyi(e, delta, N, avg=False)
#                 x_sigma2, y_sigma2 = noise_params(eps_0, delta_0, sensitivity, d)
#                 X_priv, y_priv = privatize(X, y, beta, x_sigma2, y_sigma2, e)
#             if privll:
#                 params = optimize(X_priv, y_priv, math.sqrt(y_sigma2), x_sigma2)
#             else:
#                 params = run_probit(X_priv, y_priv)
#             score = predict(params, X_test, y_test)
#             print(f"eps = {e}, score = {score}")
#         dp_res.append(score)
#         budget.append((eps_0, delta_0))
#     return dp_res, budget


# local Gaussion
def run_dp_noshuffle(X, y, X_test, y_test, beta, privll:bool):
    dp_res = []
    for e in eps:
        x_sigma2, y_sigma2 = noise_params(e, delta, sensitivity, d)
        X_priv, y_priv = privatize(X, y, beta, x_sigma2, y_sigma2, e)
        if privll:
            params = optimize(X_priv, y_priv, math.sqrt(y_sigma2), x_sigma2)
        else:
            params = run_probit(X_priv, y_priv)
        score = predict(params, X_test, y_test)
        print(f"eps = {e}, score = {score}")
        dp_res.append(score)
    return dp_res

def run_binsearch(X, y, X_test, y_test, num_iter):
    binres = []
    for e in eps:
        sigma2, eps0, delta0 = binary_search(e, delta, N, sensitivity, num_iter)
        X_priv, y_priv = privatize_rr(X, y, sigma2, eps0)
        py = math.exp(e) / (math.exp(e) + 1)
        params = optimize_rr(X_priv, y_priv, py, sigma2)
        score = predict(params, X_test, y_test)
        print(f"eps = {e} for binary search, score = {score}")
        binres.append(score)
    return binres

def run_binsearch_compose(X, y, X_test, y_test, num_iter):
    binres = []
    for e in eps:
        sigma2, eps0, delta0 = binary_search(e, delta, N, sensitivity, num_iter)
        k, shuffle_budget = compose(eps0, delta0, e, delta, N, num_iter)
        sum_params = np.zeros(shape=(d, 1))
        for i in range(k):
            X_priv, y_priv = privatize_rr(X, y, sigma2, eps0)
            py = math.exp(e) / (math.exp(e) + 1)
            params = optimize_rr(X_priv, y_priv, py, sigma2).reshape(d, 1)
            sum_params += params
        avg_params = sum_params / (np.ones((d, 1)) * k)
        score = predict(avg_params, X_test, y_test)
        print(f"eps = {e} for binary search, score = {score}")
        binres.append(score)
    return binres


def run_dp_sgd():
    X, y = generate(N, d, beta)
    X_test, y_test = generate(N, d, beta)
    train_loader = torch_data(X, y)
    test_loader = torch_data(X_test, y_test)

    central_acc = []
    for i in range(num_eps):
        central_acc.append(dp_sgd(train_loader, test_loader, eps[i], delta, d, central=True))

    plt.plot(eps, central_acc, label='DP-SGD')
    return central_acc



nondp = []
binsearch = []

for i in range(num_trials):
    X, y = generate(N, d, beta)
    X_test, y_test = generate(N, d, beta)
    nondp.append(run_nondp(X, y, X_test, y_test))
    binsearch.append(run_binsearch(X, y, X_test, y_test, num_iter=300))

nondp_mean = np.mean(nondp)

bin_mean = np.mean(binsearch, axis=0)
bin_err = np.std(binsearch, axis=0)

plt.errorbar(eps, bin_mean, yerr=bin_err, markersize=8, capsize=10, label="Binary Search")
plt.plot(eps, (nondp_mean * np.ones(shape=num_eps)), label="Non-DP")


plt.title(f"Binary Search: n = {N}, d = {d}, δ = {delta}")
plt.xlabel('espilon')
plt.ylabel('classification error')
plt.legend()
plt.show()
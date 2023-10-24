'''
Experiments for private and nonprivate probit regression
error metric is the log likelihood of estimators
tries different values of the privacy parameter
'''

from unittest import skip
from generate_data import generate, torch_data
from mechanisms import local_budget_renyi, noise_params, predict, privatize, run_probit, sampling_params, sampling_privatize, sample_sz
from priv_opt import optimize
from sgd import dp_sgd
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

# initialize parameters
d = 5   # dimensions
N = 1000  # sample size
num_trials = 5
# error = "PRED"  # error metric: LOG-LIKE, BETA-NORM, PRED

# privacy parameters
eps = np.arange(start=1, stop=40, step=10)    # overall epsilon values after shuffling
num_eps = eps.shape[-1]
delta = 0.1
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
def run_dp_shuffle(X, y, X_test, y_test, beta, avg:bool, sampling:bool, privll:bool):
    dp_res = []
    budget = []
    for e in eps:
        if avg:
            eps_0, delta_0, k = local_budget_renyi(e, delta, N, avg=True)
            avg_params = averaging(X, y, eps_0, delta_0, k, beta, privll)
            score = predict(avg_params, X_test, y_test)
            print(f"eps = {e}, score = {score}")
        else:
            if sampling:
                sz = sample_sz(N, e, delta)
                y_sigma2, x_sigma2 = sampling_params(sz, d, e, delta, sensitivity)
                X_priv, y_priv = sampling_privatize(X, y, beta, y_sigma2, x_sigma2, sz)
                #print(f"subsampling noise for y = {y_sigma2}")
            else:
                eps_0, delta_0, k = local_budget_renyi(e, delta, N, avg=False)
                x_sigma2, y_sigma2 = noise_params(eps_0, delta_0, sensitivity, d)
                X_priv, y_priv = privatize(X, y, beta, x_sigma2, y_sigma2, e)
            if privll:
                params = optimize(X_priv, y_priv, math.sqrt(y_sigma2), x_sigma2)
            else:
                params = run_probit(X_priv, y_priv)
            score = predict(params, X_test, y_test)
            print(f"eps = {e}, score = {score}")
        dp_res.append(score)
        budget.append((eps_0, delta_0))
    return dp_res, budget


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

def shuffle_trials(X, y, X_test, y_test, beta):
    nondp_res = run_nondp(X, y, X_test, y_test)
    shuffle_res, budget = run_dp_shuffle(X, y, X_test, y_test, beta, avg=False, sampling=False, privll=True)
    print(f"non-DP = {nondp_res} \n", f"DP = {shuffle_res}")
    return nondp_res, shuffle_res

def localdp_trials(X, y, X_test, y_test, beta):
    probit_res = run_dp_noshuffle(X, y, X_test, y_test, beta, privll=False)
    dp_res = run_dp_noshuffle(X, y, X_test, y_test, beta, privll=True)
    print(f"Local with Probit = {probit_res} \n", f"Local with DP LL = {dp_res}")
    return probit_res, dp_res

def ll_trials(X, y, X_test, y_test, beta):
    probit_ll, budget = run_dp_shuffle(X, y, X_test, y_test, beta, avg=False, sampling=False, privll=False)
    dp_ll, _ = run_dp_shuffle(X, y, X_test, y_test, beta, avg=False, sampling=False, privll=True)
    print(f"probit log-likelihood = {probit_ll} \n", f"DP log-likelihood = {dp_ll}")
    return probit_ll, dp_ll, budget

def avg_trials(X, y, X_test, y_test, beta):
    #nonavg = run_dp(X, y, X_test, y_test, beta, avg=False, sampling=False, privll=True)
    #samp = run_dp(X, y, X_test, y_test, beta, avg=False, sampling=True, privll=True)
    avg = run_dp_shuffle(X, y, X_test, y_test, beta, avg=True, sampling=False, privll=True)
    print(f"averaging = {avg}")
    return avg

def samp_trials(X, y, X_test, y_test, beta):
    samp = run_dp_shuffle(X, y, X_test, y_test, beta, avg=False, sampling=True, privll=True)
    #print(f"sampling = {samp}")
    return samp

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
probitll = []
dpll = []
avg = []
samp = []
shuffle = []
local_dp = []
local_prob = []

for i in range(num_trials):
    X, y = generate(N, d, beta)
    X_test, y_test = generate(N, d, beta)
    nondp.append(run_nondp(X, y, X_test, y_test))
    #nondp, dp = shuffle_trials(X, y, X_test, y_test, beta)
    localprob, localdp = localdp_trials(X, y, X_test, y_test, beta)
    #probitll_res, dpll_res, budget = ll_trials(X, y, X_test, y_test, beta)
    #avg_res = avg_trials(X, y, X_test, y_test, beta)
    #samp_res = samp_trials(X, y, X_test, y_test, beta)

    print(f'''round {i}
        Non-DP: {max(nondp)},
        Local Gaussian with Probit: {max(localprob)},
        Local Gaussian with Custom LL: {max(localdp)},
    ''')

    local_dp.append(localdp)
    local_prob.append(localprob)
    #shuffle.append(dp_shuffle)
    # probitll.append(probitll_res)
    # dpll.append(dpll_res)
    #avg.append(avg_res)
    #samp.append(samp_res)

nondp_mean = np.mean(nondp)
#nondp_err = np.std(nondp, axis=0)

localdp_mean = np.mean(local_dp, axis=0)
localdp_err = np.std(local_dp, axis=0)

localprob_mean = np.mean(local_prob, axis=0)
localprob_err = np.std(local_prob, axis=0)

# shuffle_mean = np.mean(shuffle, axis=0)
# shuffle_err = np.std(shuffle, axis=0)

# probit_mean = np.mean(probitll, axis=0)
# probit_err = np.std(probitll, axis=0)

# dp_mean = np.mean(dpll, axis=0)
# dp_err = np.std(dpll, axis=0)

# avg_mean = np.mean(avg, axis=0)
# avg_err = np.std(avg, axis=0)

# samp_mean = np.mean(samp, axis=0)
# samp_err = np.std(samp, axis=0)
# results = {'eps': eps, 'local budget': budget, 'DP score': dp_mean, 'Probit score': probit_mean}
# df = pd.DataFrame(results)
# print(df)

plt.errorbar(eps, localdp_mean, yerr=localdp_err, markersize=8, capsize=10, label="Local DP with Custom")
plt.errorbar(eps, localprob_mean, yerr=localprob_err, markersize=8, capsize=10, label="Local DP with Probit")
#plt.errorbar(eps, shuffle_mean, yerr=shuffle_err, markersize=8, capsize=10, label="Shuffling")
# plt.errorbar(eps, probit_mean, yerr=probit_err, markersize=8, capsize=10, label="Probit LL")
# plt.errorbar(eps, dp_mean, yerr=dp_err, markersize=8, capsize=10, label="DP LL")
#plt.errorbar(eps, avg_mean, yerr=avg_err, markersize=8, capsize=10, label="Averaging")
#plt.errorbar(eps, samp_mean, yerr=samp_err, markersize=8, capsize=10, label="Sampling")
plt.plot(eps, (nondp_mean * np.ones(shape=num_eps)), label="Non-DP")

#central_acc = run_dp_sgd()

# print(f'''Experiment results:
#     non-privatized: {max(nondp)}
#     probit log-like: {max(probitll)}
#     private log-like: {max(dpll)}
#     averaging: {max(avg)}
#     subsampling: {max(samp)}
# ''')


plt.title(f"Private X and Y: n = {N}, d = {d}")
plt.xlabel('espilon')
plt.ylabel('classification error')
plt.legend()
plt.show()
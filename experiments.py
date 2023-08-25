'''
Experiments for private and nonprivate probit regression
error metric is the log likelihood of estimators
tries different values of the privacy parameter
'''

from generate_data import generate, torch_data
from mechanisms import local_budget, predict, probit_err, privatize, run_probit
from sgd import dp_sgd
import numpy as np
import math
import matplotlib.pyplot as plt

# initialize parameters
K = 10   # dimensions
N = 400  # sample size
num_trials = 1
error = "PRED"  # error metric: LOG-LIKE, BETA-NORM, PRED

# privacy parameters
eps = np.arange(start=0.1, stop=6, step=0.2)    # overall epsilon values after shuffling
num_eps = eps.shape[-1]
delta = 0.1
sensitivity = 2.0 # range that X values take
beta = np.random.normal(loc=0, scale=math.sqrt(math.sqrt(N)), size=K)

def run_dp(X, y, X_test, y_test):
    dp_res = np.zeros(shape=num_eps)
    for i in range(num_eps):
        eps_0, delta_0 = local_budget(eps[i], delta, N, avg=False)
        X_priv, y_priv = privatize(X, y, K, eps_0, delta_0, sensitivity, beta)
        for j in range(num_trials):
            dp_res[i] += probit_err(X_priv, y_priv, error, N, K, beta, X_test, y_test)
        dp_res[i] /= num_trials
    return dp_res

def run_nondp(X, y, X_test, y_test):
    nondp_res = []
    m = num_trials * num_eps
    for i in range(m):
        nondp_res.append(probit_err(X, y, error, N, K, beta, X_test, y_test))
    return max(nondp_res)

def avg_params(X, y, eps_0, delta_0):
    m = math.ceil(math.sqrt(N))
    sum_params = np.zeros(shape=(K, 1))
    for i in range(m):
        X_priv, y_priv = privatize(X, y, K, eps_0, delta_0, sensitivity, beta)
        _, _, params = run_probit(X_priv, y_priv)
        #print("sigma 2", sigma2)
        #params = label_priv_reg(X_priv, y_priv, math.sqrt(sigma2))
        #print("params", params)
        sum_params += params
    avg_params = sum_params / (np.ones((K, 1)) * m)
    print("AVG PARAMS: ", avg_params)
    return avg_params

def run_trials(X, y, X_test, y_test):
    nondp_res = run_nondp(X, y, X_test, y_test)
    dp_res = run_dp(X, y, X_test, y_test)

    plt.style.use('dark_background')
    plt.plot(eps, dp_res, label='Local DP')
    plt.plot(eps, (nondp_res * np.ones(shape=num_eps)), label='non-DP')
    return nondp_res, dp_res

def run_averaged_trials(X, y, X_test, y_test):
    avg_dp_res = np.zeros(shape=num_eps)
    for i in range(num_eps):
        for j in range(num_trials):
            eps_0, delta_0 = local_budget(eps[i], delta, N, avg=True)
            dp_params = avg_params(X, y, eps_0, delta_0)
            avg_dp_res[i] += predict(dp_params, N, X_test, y_test)
        avg_dp_res[i] /= num_trials

    plt.style.use('dark_background')
    plt.plot(eps, avg_dp_res, label='Local DP (averaged)')
    return avg_dp_res

def run_dp_sgd():
    X, y = generate(N, K, beta)
    X_test, y_test = generate(N, K, beta)
    train_loader = torch_data(X, y)
    test_loader = torch_data(X_test, y_test)

    central_acc = []
    for i in range(num_eps):
        central_acc.append(dp_sgd(train_loader, test_loader, eps[i], delta, K, central=True))

    plt.plot(eps, central_acc, label='DP-SGD')
    return central_acc



X, y = generate(N, K, beta)
X_test, y_test = generate(N, K, beta)

nondp_res, dp_res = run_trials(X, y, X_test, y_test)
avg_dp_res = run_averaged_trials(X, y, X_test, y_test)
#central_acc = run_dp_sgd()

print("NONDP RESULT: ", nondp_res)
print("DP RESULT: ",dp_res)
print("DP WITH AVERAGING: ", avg_dp_res)

plt.title(f"n = {N}, d = {K}")
plt.xlabel('espilon')
plt.ylabel('classification error')
plt.legend()
plt.show()

#print("DP-SGD RESULT: ", central_acc)

# print("BETA:", beta)
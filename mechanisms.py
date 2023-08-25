'''
baseline mechanisms for differentially private and nonprivate probit regression
'''

import numpy as np
import statsmodels.api as smf
import math
import pandas as pd
from sklearn.metrics import accuracy_score

DELTA1 = 1e-5    # intermediate delta for shuffling, delta \in [0,1]
                 # smaller values = poorer performance
DELTA2 = 1e-5    # intermediate delta for k-composition


def predict(beta_hat, N, X_test, y_test):
    y_pred = np.empty(shape=N)
    for i in range(N):
        rho = np.random.standard_normal()
        xbeta = np.dot(X_test[i], beta_hat)
        if xbeta + rho > 0:
            y_pred[i] = 1
        else:
            y_pred[i] = 0
    return accuracy_score(y_test, y_pred)

def run_probit(X, y):
    probit_model=smf.Probit(y,X)
    result=probit_model.fit()
    #print(result.summary2())
    params = pd.DataFrame(result.params,columns={'coef'},).to_numpy()
    #print(probit_model.hessian(params))
    return probit_model, result, params

# runs probit regression and returns error metric
def probit_err(X, y, error, N, K, beta, X_test, y_test):
    _, result, params = run_probit(X, y)
    print(params)
    if error == "LOG-LIKE":
        return result.llf / N
    elif error == "BETA-NORM":
        print(np.linalg.norm(beta - params.T))
        return np.linalg.norm(beta - params.T)
    elif error == "PRED":
        return predict(params, N, X_test, y_test)


def priv_params(K, sensitivity, eps, delta, beta):
    x_sigma2 = ((K*sensitivity)**2/eps**2)*(2.0*math.log(1.25/delta))
    gamma = 1/math.sqrt(1+(x_sigma2*np.linalg.norm(np.square(beta))))
    y_sigma2 = (1/eps**2)*(2.0*math.log(1.25/delta))    # sensitivity of y's = 1
    return x_sigma2, gamma, y_sigma2

# add Gaussian noise to X's and y's
def privatize(X, y, K, eps, delta, sensitivity, beta):
    N = np.shape(y)[0]
    x_sigma2, gamma, y_sigma2 = priv_params(K, sensitivity, eps, delta, beta)
    
    X_priv = np.copy(X)
    y_priv = np.zeros(N)

    for i in range(N):
        x_noise = np.random.normal(loc=0, scale=math.sqrt(x_sigma2), size=K)
        X_priv[i] += x_noise
        X_priv[i] *= gamma

        y_noise = np.random.normal(loc=0, scale=math.sqrt(y_sigma2))
        if y_noise + float(y[i]) >= 0.5:
            y_priv[i] += 1

    return X_priv, y

# returns (ε_0, δ_0) for local randomizer
def local_budget(eps, delta, n, avg: bool):
    if avg:
        k = math.log(math.sqrt(n))
        eps_comp_factor = (math.sqrt(2) * (math.sqrt(k*math.log(1/DELTA2) + 2*k) 
                                    - math.sqrt(k*math.log(1/DELTA2)))) / (2*k)
        eps *= eps_comp_factor
        old_eps = eps*math.sqrt(2*k*math.log(1/DELTA2)) + (k*eps*(math.exp(eps)-1))
        delta = (delta - DELTA2) / k
        print(f"after averaging: eps = {old_eps}, eps' = {eps}, delta' = {delta}")

    eps_shuffle_factor = 2*math.log( (math.sqrt(n + 4*math.log(1/delta)) + math.sqrt(n)) / (2*math.log(1/delta)) )
    eps_0 = eps * eps_shuffle_factor
    delta_0 = (delta - DELTA1) / (n * (math.exp(eps) + 1) * (1 + math.exp(-1 * eps_0/2)))

    if avg:
        print(f"(ε_0, δ_0) for averaged trials = ({eps_0}, {delta_0})")
    else:
        print(f"(ε_0, δ_0) without averaging = ({eps_0}, {delta_0})")
    
    return eps_0, delta_0
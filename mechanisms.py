'''
baseline mechanisms for differentially private and nonprivate probit regression
'''

import numpy as np
import statsmodels.api as smf
import math
import pandas as pd
from sklearn.metrics import accuracy_score

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
    return probit_model, result, params

def nondp_loglike(X, y, X_priv):
    nondp_model, _, _ = run_probit(X, y)
    _, _, dp_params = run_probit(X_priv, y)
    return nondp_model.loglike(dp_params)

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

# run Gaussian mechanism on X's
def privatize(X, N, K, eps, delta, sensitivity, beta):
    m = math.ceil(math.sqrt(N))   # number of times we add noise to x_i

    sigma2 = ((K*sensitivity)**2/eps**2)*(2.0*math.log(1.25/delta))
    gamma = 1/math.sqrt(1+((1/m)*(sigma2**2)*np.linalg.norm(np.square(beta))))
    print("GAMMA: ", gamma)
    
    X_priv = np.copy(X)
    for i in range(N):
        noise = 0
        noise += np.random.normal(loc=0, scale=math.sqrt(sigma2), size=(m,K))
        noise = np.sum(noise, axis=0) / (np.ones(shape = K) * m)
        X_priv[i] += noise
        X_priv[i] *= gamma

    print(X_priv)
    return X_priv
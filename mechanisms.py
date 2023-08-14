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

# add Gaussian noise to X's and y's
def privatize(X, y, N, K, eps, delta, sensitivity, beta):
    sigma2 = ((K*sensitivity)**2/eps**2)*(2.0*math.log(1.25/delta))
    gamma = 1/math.sqrt(1+((sigma2**2)*np.linalg.norm(np.square(beta))))
    y_sigma2 = (1/eps**2)*(2.0*math.log(1.25/delta))    # sensitivity of y's = 1
    
    X_priv = np.copy(X)
    y_priv = np.zeros(N)

    for i in range(N):
        x_noise = np.random.normal(loc=0, scale=math.sqrt(sigma2), size=K)
        X_priv[i] += x_noise
        X_priv[i] *= gamma

        y_noise = np.random.normal(loc=0, scale=math.sqrt(y_sigma2))
        if y_noise + float(y[i]) >= 0.5:
            y_priv[i] += 1

    return X_priv, y_priv
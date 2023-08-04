'''
baseline mechanisms for differentially private and nonprivate probit regression
'''

import numpy as np
import statsmodels.api as smf
import math
import pandas as pd
from sklearn.metrics import accuracy_score
from scipy import stats

from generate_data import generate

def predict(beta_hat, N, K, beta):
    X_test, y_test = generate(N, K, beta)
    y_pred = np.empty(shape=N)
    for i in range(N):
        rho = np.random.standard_normal()
        xbeta = np.dot(X_test[i], beta_hat)
        if xbeta + rho > 0:
            y_pred[i] = 1
        else:
            y_pred[i] = 0
    return accuracy_score(y_test, y_pred)

# runs probit regression and returns error metric
def probit_reg(X, y, error, N, K, beta):
    probit_model=smf.Probit(y,X)
    result=probit_model.fit()
    #print(result.summary2())
    params = pd.DataFrame(result.params,columns={'coef'},).to_numpy()
    print(params)
    if error == "LOG-LIKE":
        return result.llf / N
    elif error == "BETA-NORM":
        print(np.linalg.norm(beta - params.T))
        return np.linalg.norm(beta - params.T)
    elif error == "PRED":
        return predict(params, N, K, beta)

# run Gaussian mechanism on X's
def gauss_mech(X, N, K, eps, delta, sensitivity):
    sigma2 = (sensitivity**2/eps**2)*(2.0*math.log(1.25/delta))
    X_priv = np.copy(X)
    for i in range(N):
        noise = np.random.normal(loc=0, scale=math.sqrt(sigma2), size=K)
        X_priv[i] += noise
    return X_priv
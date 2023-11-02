'''
implementing probit regression
'''

import numpy as np
import statsmodels.api as smf
import pandas as pd
from sklearn.metrics import accuracy_score


def predict(beta_hat, X_test, y_test):
    n = np.shape(y_test)[0]
    y_pred = np.empty(shape=n)
    for i in range(n):
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
    return params
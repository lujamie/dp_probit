'''
implementing probit regression
'''

import numpy as np
import statsmodels.api as smf
import pandas as pd
from sklearn.metrics import accuracy_score
from scipy.stats import norm
import random


def predict(beta_hat, X_test, y_test):
    n = np.shape(y_test)[0]
    y_pred = np.empty(shape=n)
    for i in range(n):
        xbeta = np.dot(X_test[i], beta_hat)
        py = norm.cdf(xbeta)
        if random.uniform(0, 1) < py:
            y_pred[i] = 1
        else:
            y_pred[i] = 0
    return accuracy_score(y_test, y_pred)

def run_probit(X, y):
    probit_model=smf.Probit(y,X)
    result=probit_model.fit()
    print(result.summary2())
    params = pd.DataFrame(result.params,columns={'coef'},).to_numpy()
    #print(probit_model.hessian(params))
    print("probit param:", params)
    return params
'''
calculate and plot standard errors of private and nonprivate estimators
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import statsmodels.api as smf
import math
from mechanisms import privatize, priv_params
from generate_data import generate

n = 400
d = 5
sensitivity = 2.0 # range that X values take

def fisher(X, y, beta):
    probit_model=smf.Probit(y,X)
    probit_fisher = probit_model.hessian(beta) * (-1)
    #print(f"fisher matrix from statsmodel = {probit_fisher}")
    # fisher_info = np.zeros(shape=(d, d))
    # for i in range(n):
    #     xbeta = np.dot(X[i], beta)
    #     cdf = min(norm.cdf(xbeta), 0.9999)
    #     fisher_info += np.multiply(np.outer(X[i], X[i].T), (norm.pdf(xbeta)**2 / (cdf * (1-cdf))))
    # print(f"fisher = {fisher_info}")
    return probit_fisher

def dp_fisher(X_priv, beta, eps, delta, sensitivity):
    _, _, y_sigma2 = priv_params(d, sensitivity, eps, delta, beta)
    y_sigma = math.sqrt(y_sigma2)
    dp_fisher_info = np.zeros(shape=(d, d))
    for i in range(n):
        dp_xbeta = np.dot(X_priv[i], beta)
        cdf = min(norm.cdf(dp_xbeta), 0.999999)
        dp_yprob = norm.cdf(0.5/y_sigma) * cdf + (1-norm.cdf(0.5/y_sigma)) * (1-cdf)
        dp_fisher_info += np.multiply(((2*norm.cdf(0.5/y_sigma) - 1)**2 / dp_yprob + 
                            (1- 2*norm.cdf(0.5/y_sigma))**2 / (1- dp_yprob)) * 
                            norm.pdf(dp_xbeta)**2, np.outer(X_priv[i], X_priv[i].T))
    #print(f"fisher = {dp_fisher_info}")
    return dp_fisher_info

def std_err(fisher, dp_fisher):
    vars = np.diag(fisher)
    errs = np.sqrt(np.reciprocal(vars))
    dp_vars = np.diag(dp_fisher)
    dp_errs = np.sqrt(np.reciprocal(dp_vars))
    diff = np.linalg.norm(np.absolute(dp_errs - errs))
    return diff

beta = np.random.normal(loc=0, scale=math.sqrt(math.sqrt(n)), size=d)
X, y = generate(n, d, beta)
eps = np.arange(start=0.1, stop=6, step=0.2)
delta = 1e-5
# X_priv, y_priv = privatize(X, y, d, 4.0, delta, sensitivity, beta)
# nonpriv = fisher(X, y, beta)
# priv = dp_fisher(X_priv, beta, 4.0, delta, sensitivity)

err = []

for e in eps:
    X_priv, y_priv = privatize(X, y, d, e, delta, sensitivity, beta)
    fisher_info = fisher(X, y, beta)
    dp_fisher_info = dp_fisher(X_priv, beta, e, delta, sensitivity)
    err.append(std_err(fisher_info, dp_fisher_info))

#plt.style.use('dark_background')
plt.title(f"Standard error of estimators for n = {n}, d = {d}")
plt.xlabel("epsilon")
plt.ylabel("||nonDP err - DP err||")
plt.plot(eps, err)
plt.show()
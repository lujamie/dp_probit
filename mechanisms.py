'''
baseline mechanisms for differentially private and nonprivate probit regression
'''

from threading import local
import numpy as np
import statsmodels.api as smf
import math
import pandas as pd
import random
from sklearn.metrics import accuracy_score

DELTA1 = 1e-5    # intermediate delta for shuffling, delta \in [0,1]
                 # smaller values = poorer performance
DELTA2 = 1e-5    # intermediate delta for k-composition


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


def noise_params(eps, delta, sensitivity, d):
    eps /= 2
    x_sigma2 = ((d*sensitivity)**2/eps**2)*(2.0*math.log(1.25/delta))
    y_sigma2 = (1/eps**2)*(2.0*math.log(1.25/delta))    # sensitivity of y's = 1
    return x_sigma2, y_sigma2

def rr(eps, yi):
    # wp 1-p, output true answer
    # otherwise, output 1 or 0 with even chance
    p = 2 / (math.exp(eps) + 1)
    if (random.uniform(0, 1) < p):
        if (np.random.randint(2) == 1):
            return 1
        else:
            return 0
    else:
        return yi


# add Gaussian noise to X's and y's
def privatize(X, y, beta, x_sigma2, y_sigma2):
    n = np.shape(y)[0]
    d = np.shape(X)[1]
    gamma = 1/math.sqrt(1+(x_sigma2*np.linalg.norm(np.square(beta))))
    
    X_priv = np.copy(X)
    y_priv = np.zeros(n)

    for i in range(n):
        x_noise = np.random.normal(loc=0, scale=math.sqrt(x_sigma2), size=d)
        X_priv[i] += x_noise
        #X_priv[i] *= gamma

        y_noise = np.random.normal(loc=0, scale=math.sqrt(y_sigma2))
        if y_noise + float(y[i]) >= 0.5:
            y_priv[i] += 1

    return X_priv, y_priv

# add Gaussian noise to X's and y's
def privatize_rr(X, y, beta, x_sigma2, eps):
    n = np.shape(y)[0]
    d = np.shape(X)[1]
    
    X_priv = np.copy(X)
    y_priv = np.zeros(n)

    for i in range(n):
        x_noise = np.random.normal(loc=0, scale=math.sqrt(x_sigma2), size=d)
        X_priv[i] += x_noise

        y_priv[i] = rr(eps/2, y[i])

    return X_priv, y_priv


# returns (ε_0, δ_0) for local randomizer using renyi dp shuffle
def local_budget_renyi(eps, delta, n, avg:bool):
    alpha = max(2, n / (9 * math.exp(5 * eps)))
    print(f"alpha = {alpha}")
    s = ((math.log(1/delta) + (alpha-1)*math.log(1-(1/alpha)) - math.log(alpha)) / (alpha - 1))
    print(f"subtracting {s}")
    eps_renyi = eps - ((math.log(1/delta) + (alpha-1)*math.log(1-(1/alpha)) - math.log(alpha)) / (alpha - 1))
    print(f"ε(lambda) = {eps_renyi}")
    delta_renyi = (1 - (1/alpha))**alpha * math.exp((alpha - 1)*(eps_renyi - eps)) / (alpha - 1)
    print(f"δ(lambda) = {delta_renyi}")
    if avg:
        eps_0 = eps
        k = math.ceil(eps_renyi / ((1/(alpha-1))*math.log(1 + math.comb(alpha, 2) * 4 * (math.exp(eps_0)-1)**2 / n)))
        print(f"number of rounds for averaging = {k}")
        delta_0 = max(1e-13, delta - delta_renyi) / (k*n*(math.exp(eps) + 1)*(1 + math.exp(-1 * eps_0)/2))
        print(f"(ε_0, δ_0) for averaged trials = ({eps_0}, {delta_0})")
    else:
        eps_0 = math.log(math.sqrt(n * (math.exp((alpha-1)*(eps_renyi)) - 1) / (math.comb(alpha, 2) * 4)) + 1)
        delta_0 = (max(1e-13, delta - delta_renyi)) / (n*(math.exp(eps) + 1)*(1 + math.exp(-1 * eps_0)/2))
        k = 1
        print(f"(ε_0, δ_0) without averaging = ({eps_0}, {delta_0})")    
    return eps_0, delta_0, k


local_budget_renyi(eps=5, delta=0.1, n=400, avg=False)

# return sampling size
def sample_sz(n, eps, delta):
    m = 1
    km = 0
    for k in range(n**2):
        fac = m*((math.exp(eps) - 1) / n)
        #k = eps * math.log(1 + fac)
        delta_p = delta - (k*delta/(n*math.log(n)))
        if delta_p < 0:
            break
        if math.log(1 + fac) * math.sqrt(2*k*math.log(1/delta_p)) + k*math.log(1 + fac)*fac > eps:
            continue
        if k * m > km:
            km = k*m
    print(f"sampling size = {km}")
    return km

# privatization for subsampling method
def sampling_params(n, d, eps, delta, sensitivity):
    half_delta = delta / 2
    half_eps = eps / 2
    y_sigma2 = n**2 * math.log(1/half_delta) * math.log(1/half_delta) / half_eps**2
    x_sigma2 = d * sensitivity**2 * y_sigma2
    return y_sigma2, x_sigma2

def sampling_privatize(X, y, beta, y_sigma2, x_sigma2, sz):
    n = np.shape(X)[0]
    d = np.shape(X)[1]
    sample = np.random.randint(0, n-1, size=sz)
    X_sample = np.empty((sz, d))
    y_sample = np.empty(sz)
    for i in range(sz):
        indx = sample[i]
        X_sample[i] = X[indx]
        y_sample[i] = y[indx] 
    X_priv, y_priv = privatize(X_sample, y_sample, beta, x_sigma2, y_sigma2)
    return X_priv, y_priv
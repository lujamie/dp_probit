import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

dtype = tf.float32

# private log-likelihood
def nll_priv(beta, X_priv, Y_priv, y_sigma, x_sigma2):
    p = tfd.Normal(loc=0., scale=1.)
    cdf_z = p.cdf(0.5/y_sigma)
    gamma = 1/math.sqrt(1+(x_sigma2*tf.norm(tf.square(beta), ord=1)))
    mu = tf.linalg.matvec(tf.multiply(gamma, X_priv), beta)
    cdf_mu = tf.map_fn(p.cdf, mu)
    cdf_mu_inv = tf.math.subtract(1, cdf_mu)
    py = tf.math.scalar_mul(cdf_z, cdf_mu) + tf.math.scalar_mul(1-cdf_z, cdf_mu_inv)
    ll = tf.math.reduce_sum(tf.math.multiply(Y_priv, tf.math.log(py)) + 
            tf.math.multiply(tf.math.subtract(1, Y_priv), tf.math.log(tf.math.subtract(1, py))))
    return -1 * ll

def nll_wrapper(X_priv, Y_priv, y_sigma, x_sigma2):
    return (lambda b: nll_priv(b, X_priv, Y_priv, y_sigma, x_sigma2))

# return function value and gradients
def neg_like_and_gradient(params, X_priv, Y_priv, y_sigma, x_sigma2):
    return tfp.math.value_and_gradient(nll_wrapper(X_priv, Y_priv, y_sigma, x_sigma2), params)

def grad_wrapper(X_priv, Y_priv, y_sigma, x_sigma2):
    return (lambda b: neg_like_and_gradient(b, X_priv, Y_priv, y_sigma, x_sigma2))

# run private optimization
def optimize(x_priv, y_priv, y_sigma, x_sigma2):
    X_priv = tf.constant(x_priv, dtype=dtype)
    Y_priv = tf.constant(y_priv, dtype=dtype)
    # set some naiive starting values
    d = np.shape(x_priv)[1]
    start = [0.] * d
    # optimization
    optim_results = tfp.optimizer.bfgs_minimize(
        grad_wrapper(X_priv, Y_priv, y_sigma, x_sigma2), start, tolerance=1e-6)

    # organize results
    est_params = optim_results.position.numpy()
    est_serr = np.sqrt(np.diagonal(optim_results.inverse_hessian_estimate.numpy()))
    #print(pd.DataFrame(np.c_[est_params, est_serr],columns=['estimate', 'std err']))
    return est_params


'''
given total privacy budget (eps, delta) computes min(sigma2) with shuffling bounds
'''

import math

def binary_search(localparams, eps, delta, n, gst, num_iterations):
    '''
    binary search to find min sigma s.t. exists (eps0, delta0) -> (eps, delta) thru shuffling
    localparams = function that takes sigma as input and True if we can find such (eps0, delta0)
    eps, delta = goal budget
    n = dataset size
    gst = sensitivity (range of X values)
    num_iterations = number of iterations, accuracy is 2^(-num_iterations)*sigmaupper
    '''
    sigma2upper = (gst/eps)**2 * 2 * math.log(1.25 / delta)
    llim = 0
    rlim = sigma2upper
    eps0 = eps
    delta0 = delta
    for t in range(num_iterations):
        midsigma2 = (rlim + llim) / 2
        print(f"rlim = {rlim}")
        print(f"llim = {llim}")
        print(f"midsigm2  = {midsigma2}")
        valid, neweps0, newdelta0 = localparams(midsigma2, eps, delta, n, gst)
        if valid:
            print("valid!")
            llim = llim
            rlim = midsigma2
            eps0 = neweps0
            delta0 = newdelta0
        else:
            llim = midsigma2
            rlim = rlim
    return rlim, eps0, delta0

def localparams(sigma2, eps, delta, n, gst):
    '''
    gst = sensitivity, sigma2 = current noise in binary search
    initialize delta0 = delta/n, increment by 1/10n until = delta
    solves for eps0 given delta0 and sigma2 and check shuffle params
    return eps0, delta0 that works (gets us to eps, delta with shuffling)
    '''
    print(f"current σ^2 = {sigma2}")
    delta0 = delta / (4 * n * (1 + math.exp(eps)))
    while delta0 < delta:
        print(f"δ_0 = {delta0}")
        eps0 = math.sqrt((gst**2/sigma2) * 2 * math.log(1.25 / delta0))
        print(f"ε_0 = {eps0}")
        if eps0 <= 0:
            delta0 *= 10
            continue
        a = (math.exp(eps) + 1) * (1 + (1 / (math.exp(eps0) * 2)))
        deltap = delta - (a * n * delta0)
        print(f"δ' = {deltap}")
        if deltap <= 0:
            break
        b = math.log((n / (8 * math.log(2/deltap))) - 1)
        if eps0 > b:
            delta0 *= 10
            continue
        print("valid ε_0 for shuffling")
        c = 4 * math.sqrt(2 * math.log(2/deltap)) / (math.sqrt(math.exp(eps0) + 1) * n)
        eps_shuffle = math.log(1 + ((math.exp(eps0) - 1) * (c + 4/n)))
        print(f"ε for shuffling with {eps0} = {eps_shuffle}")
        if eps_shuffle > eps:
            delta0 *= 10
            continue
        return True, eps0, delta0
    return False, eps, delta

def shuffleamp(eps0, delta0, eps, delta, n):
    if eps0 == eps:
        return eps
    a = (math.exp(eps) + 1) * (1 + (1 / (math.exp(eps0) * 2)))
    deltap = delta - (a * n * delta0)
    c = 4 * math.sqrt(2 * math.log(2/deltap)) / (math.sqrt(math.exp(eps0) + 1) * n)
    eps_shuffle = math.log(1 + ((math.exp(eps0) - 1) * (c + 4/n)))
    return eps_shuffle

n = 10000
d = 5
eps = 3
delta = 1/n
gst = 2 * d
num_iterations = 300

sigma2, eps0, delta0 = binary_search(localparams, eps, delta, n, gst, num_iterations)
eps_shuffle = shuffleamp(eps0, delta0, eps, delta, n)
sigma2upper = (gst/eps)**2 * 2 * math.log(1.25 / delta)
print(f'''Given (ε, δ) = ({eps}, {delta}), the original noise σ^2 = {sigma2upper}.
With shuffling, we can use (ε_0, δ_0) = ({eps0}, {delta0}) 
    so ε_shuffle = {eps_shuffle} and σ^2 = {sigma2}.''')
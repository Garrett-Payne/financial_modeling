## This script is intended to show procesing times for function to calculate binomial trees
## Taken from chapter 10, pages 293 - 298 of Reference book
import time
import numpy as np
import numba
import math
import pyximport; pyximport.install(pyimport=True,language_level=3)

# intitialize some variable
S0 = 36. # value of asset
T=1.0 # time horizon for model
r=0.06  # short rate (set as constant)
sigma = 0.2 # volatility factor

## Note: This code is intended to model the binomial option price model, created by Cox-Ross-Rubinstein

## create pythonic iterative function
def simulate_tree(M): 
    dt=T/M
    u = math.exp(sigma * math.sqrt(dt)) 
    d=1/u 
    S=np.zeros((M+1,M+1)) 
    S[0,0]=S0
    z=1
    for t in range(1, M + 1):
        for i in range(z): 
            S[i,t]=S[i,t-1]*u 
            S[i+1, t] = S[i, t-1] * d
        z+=1 
    return S

n = 4
t1 = time.time()
tn = simulate_tree(n)
t2 = time.time()
print("Time to run pythonic binomial model with {} iterations: {} secs".format(n,t2-t1))

n = 500
t1 = time.time()
tn = simulate_tree(n)
t2 = time.time()
print("Time to run pythonic binomial model with {} iterations: {} secs\n".format(n,t2-t1))

## vectorization using numpy

def simulate_tree_np(M): 
    dt=T/M
    up = np.arange(M + 1)
    up = np.resize(up, (M + 1, M + 1))
    down = up.transpose() * 2
    S = S0 * np.exp(sigma * math.sqrt(dt) * (up - down))
    return S

n = 4
t1 = time.time()
tn = simulate_tree_np(n)
t2 = time.time()
print("Time to run numpy binomial model with {} iterations: {} secs".format(n,t2-t1))

n = 500
t1 = time.time()
tn = simulate_tree_np(n)
t2 = time.time()
print("Time to run numpy binomial model with {} iterations: {} secs\n".format(n,t2-t1))

## generate function using Numba
simulate_tree_nb = numba.jit(simulate_tree)

n = 4
t1 = time.time()
tn = simulate_tree_nb(n)
t2 = time.time()
print("Time to run numba pythonic binomial model with {} iterations: {} secs".format(n,t2-t1))

n = 500
t1 = time.time()
tn = simulate_tree_nb(n)
t2 = time.time()
print("Time to run numpy pythonic model with {} iterations: {} secs\n".format(n,t2-t1))

## test using cython-dedicated pyx function
import cython_binomial

n = 4
t1 = time.time()
tn = cython_binomial.simulate_tree_cy(n)
t2 = time.time()
print("Time to run numba pythonic binomial model with {} iterations: {} secs".format(n,t2-t1))

n = 500
t1 = time.time()
tn = cython_binomial.simulate_tree_cy(n)
t2 = time.time()
print("Time to run numpy pythonic model with {} iterations: {} secs\n".format(n,t2-t1))
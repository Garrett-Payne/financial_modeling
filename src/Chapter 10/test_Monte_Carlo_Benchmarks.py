## This script is intended to show procesing times of Monte Carlo Simulations
## Taken from chapter 10, pages 299 -304 of Reference book
import time
import numpy as np
import numba
import math
import pyximport; pyximport.install(pyimport=True,language_level=3)
import multiprocessing as mp

## Note: The intent here is to model the Black-Scholes-Merton difference Equation
# discretized version (Euler scheme) within Monte Carlo simulations. The model is defined,
# as well as the number of iterations to run the Monte Carlo sims.


# define some model parameters
M = 100 # number of time intervals to discretize
I = 50000 # number of paths to be simulated
T=1.0 # time horizon for model
S0 = 36. # value of asset
r=0.06  # short rate (set as constant)
sigma = 0.2 # volatility factor

# note: the function below only has 1 input p that's a tuple to define M & I. The rest of the
# variables in the function are defined above 

# define hybrid model - is mostly pythonic but does rely on some numpy data
def mcs_simulation_py(p): 
    M,I=p
    dt=T/M
    S = np.zeros((M + 1, I))
    S[0] = S0
    rn = np.random.standard_normal(S.shape) 
    for t in range(1, M + 1):
        for i in range(I): 
            S[t,i]=S[t-1,i]*math.exp((r-sigma**2/2)*dt+sigma * math.sqrt(dt) * rn[t, i])
    return S

t1 = time.time()
S = mcs_simulation_py((M, I))
t2 = time.time()
print("Time to run hybrid model: {} secs".format(t2-t1))
print("Mean end-of-period value from model: {}".format(S[-1].mean()))
print("Theoretical end-of-period value: {}".format(S0 * math.exp(r * T)))

# define strike price of model
K = 40.

C0 = math.exp(-r * T) * np.maximum(K - S[-1], 0).mean() 
print("Monte Carlo Estimator for given option of K = {}: {}\n".format(K,C0))

## create full-numpy function
def mcs_simulation_np(p): 
    M,I=p
    dt=T/M
    S = np.zeros((M + 1, I))
    S[0] = S0
    rn = np.random.standard_normal(S.shape) 
    for t in range(1, M + 1):
        S[t]=S[t-1]*np.exp((r-sigma**2/2)*dt+ sigma * math.sqrt(dt) * rn[t])
    return S

t1 = time.time()
S = mcs_simulation_np((M, I))
t2 = time.time()
print("Time to run numpy model: {} secs".format(t2-t1))
print("Mean end-of-period value from model: {}\n".format(S[-1].mean()))

## using Numba
mcs_simulation_nb = numba.jit(mcs_simulation_py)

t1 = time.time()
S = mcs_simulation_nb((M, I))
t2 = time.time()
print("Time to run Numba model: {} secs".format(t2-t1))
print("Mean end-of-period value from model: {}\n".format(S[-1].mean()))

## test using Cython function (from dedicated pyx script)
import cython_Monte_Carlo

t1 = time.time()
S = cython_Monte_Carlo.mcs_simulation_cy((M, I))
t2 = time.time()
print("Time to run Numba model: {} secs".format(t2-t1))
print("Mean end-of-period value from model: {}\n".format(S[-1].mean()))

## Using Multi-Processing
if __name__ == '__main__':
    # define pool object with 4 processes
    pool = mp.Pool(processes=4)

    # set number of chunks for simulation
    p = 20

    # using numpy function
    t1 = time.time()
    S = np.hstack(pool.map(mcs_simulation_np,p * [(M, int(I / p))]))
    t2 = time.time()
    print("Time to run multiprocessing Numpy model: {} secs".format(t2-t1))
    print("Mean end-of-period value from model: {}\n".format(S[-1].mean()))

    # using numba function
    t1 = time.time()
    S = np.hstack(pool.map(mcs_simulation_nb,p * [(M, int(I / p))]))
    t2 = time.time()
    print("Time to run multiprocessing numba model: {} secs".format(t2-t1))
    print("Mean end-of-period value from model: {}\n".format(S[-1].mean()))

        # using cython function
    t1 = time.time()
    S = np.hstack(pool.map(cython_Monte_Carlo.mcs_simulation_cy,p * [(M, int(I / p))]))
    t2 = time.time()
    print("Time to run multiprocessing cython model: {} secs".format(t2-t1))
    print("Mean end-of-period value from model: {}\n".format(S[-1].mean()))

    # make sure to close out pool
    pool.close()
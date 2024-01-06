import numpy as np
cimport cython
from libc.math cimport exp, sqrt
cdef float S0 = 36.
cdef float T = 1.0
cdef float r = 0.06
cdef float sigma = 0.2
def simulate_tree_cy(int M):
    cdef int z,t,i
    cdef float dt, u, d
    cdef float[:, :] S = np.zeros((M + 1, M + 1),dtype=np.float32)
    dt=T/M
    u = exp(sigma * sqrt(dt)) 
    d=1/u
    S[0,0]=S0
    z=1
    for t in range(1, M + 1):
        for i in range(z): 
            S[i,t]=S[i,t-1]*u 
            S[i+1, t] = S[i, t-1] * d
        z+=1
    return np.array(S)
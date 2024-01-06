import numpy as np
#cimport numpy as np
cimport cython
from libc.math cimport exp,sqrt 
cdef float S0 = 36.
cdef float T = 1.0
cdef float r = 0.06
cdef float sigma = 0.2 
@cython.boundscheck(False) 
@cython.wraparound(False)
def mcs_simulation_cy(p):
    cdef int M, I
    M,I=p
    cdef int t, i
    cdef float dt=T/M
    cdef double[:, :] S = np.zeros((M + 1, I))
    cdef double[:, :] rn = np.random.standard_normal((M + 1, I)) 
    S[0] = S0
    for t in range(1, M + 1): 
        for i in range(I):
            S[t,i]=S[t-1,i]*exp((r-sigma**2/2)*dt+ sigma * sqrt(dt) * rn[t, i])
    return np.array(S)
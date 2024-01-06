import numpy as np
cimport cython
@cython.boundscheck(False) 
@cython.wraparound(False)
def ewma_cy(double[:] x, float alpha):
    cdef int i
    cdef double[:] y = np.empty_like(x) 
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i]=alpha*x[i]+(1-alpha)*y[i-1] 
    return y
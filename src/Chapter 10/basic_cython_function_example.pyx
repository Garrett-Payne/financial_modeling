from libc.stdlib cimport rand 
cdef extern from 'limits.h':
    int INT_MAX
def average_cy2(int n):
    cdef int i 
    cdef float s=0 
    for i in range(n):
        s += rand() / INT_MAX 
    return s/n
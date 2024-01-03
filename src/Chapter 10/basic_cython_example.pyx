from libc.stdlib cimport rand
cdef extern from 'limits.h': 
    int INT_MAX
cdef int i
cdef float rn
for i in range(5):
    rn = rand() / INT_MAX
    print(rn)
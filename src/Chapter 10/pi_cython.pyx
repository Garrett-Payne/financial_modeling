## Note: I had to do these functions a little differently than defined in the book.
# I had to explicitly define the direction (magnitude z) and the check value (1) as floats
# since python kept erroring out when trying to compile the code. I assume the default compilation
# made them two different data types
import random
def mcs_pi_cy1(int n):
    cdef int i, circle = 0 
    cdef float x, y, z, chk = 1.0
    
    for i in range(n):
        x, y = random.random(), random.random() 
        z = (x**2 + y**2)**0.5

        #if ((x ** 2.0 + y ** 2.0) ** 0.5) <= 1.0:     
        if z <= chk:
            circle += 1 
    return (4 * circle) / n
  

from libc.stdlib cimport rand 
cdef extern from 'limits.h':
    int INT_MAX
def mcs_pi_cy2(int n):
    cdef int i, circle = 0 
    cdef float x, y,z, chk=1
    for i in range(n):
        x, y = rand() / INT_MAX, rand() / INT_MAX 
        z = (x**2+y**2)**0.5
        if z <= chk:
            circle += 1 
    return (4 * circle) / n
## This script is intended to show procesing times for various types of loops
## Taken from chapter 10, pages 277 - 279 of Reference book
import random
import time
import numpy as np
import numba

# Test a basic for loop
def average_py(n): 
    s=0
    for i in range(n):
        s += random.random()
    return s/n 

# define number of iterations to go through
n = 10000000 

# time how long it takes to do basic python loop
t1 = time.time()
averaged = average_py(n)
t2 = time.time()
print("Time to complete basic python for loop in dedication function, # iterations = {}, {} secs\n".format(n,t2-t1))

# time to average using for loop, not in defined function
t1 = time.time()
avg2 = sum([random.random() for _ in range(n)]) / n
t2 = time.time()
print("Time to complete basic python for loop in one line, # iterations = {}, {} secs\n".format(n,t2-t1))


## Now test with numpy

# create new fuction to average over using numpy
def average_np(n):
    s = np.random.random(n)
    return s.mean()

# now time it
t1 = time.time()
avgnp = average_np(n)
t2 = time.time()
print("Time to complete numpy average over large array, # iterations = {}, {} secs\n".format(n,t2-t1))

## test with Numba
# create the numba function
average_nb = numba.jit(average_py)

t1 = time.time()
avgnb = average_nb(n)
t2 = time.time()
print("Time to complete numba average over large array, # iterations = {}, {} secs\n".format(n,t2-t1))


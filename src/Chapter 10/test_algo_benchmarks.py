## This script is intended to show procesing times for different benchmarks
## Taken from chapter 10, pages 282 - 285 of Reference book

import time
import numba
import cython
import pyximport; pyximport.install(pyimport=True,language_level=3)
import multiprocessing as mp

# import cython script - this script will automatically compile the functions
import is_prime_cython


#############################################
######## Benchmark 1: Prime Numbers #########

## pythonic way

# make basic function
def is_prime(I): 
    if I % 2==0: return False
    for i in range(3, int(I ** 0.5) + 1, 2): 
        if I%i==0: return False
    return True

is_prime_nb = numba.jit(is_prime)


# Create test values
n = int(1e8 + 3)
p1 = int(1e8 + 7)
p2 = 100109100129162907


# Test times
t1 = time.time()
n_isp = is_prime(n)
t2 = time.time()
print("Time to complete pythonic prime function for number = {}: {} secs".format(n,t2-t1))

t1 = time.time()
n_isp = is_prime(p1)
t2 = time.time()
print("Time to complete pythonic prime function for number = {}: {} secs".format(p1,t2-t1))

t1 = time.time()
n_isp = is_prime(p2)
t2 = time.time()
print("Time to complete pythonic prime function for number = {}: {} secs\n".format(p2,t2-t1))


## using Numba

# use numba to transform and compile function for optimization
is_prime_nb = numba.jit(is_prime)

# Test times
t1 = time.time()
n_isp = is_prime_nb(n)
t2 = time.time()
print("Time to complete numba prime function for number = {}: {} secs".format(n,t2-t1))

t1 = time.time()
n_isp = is_prime_nb(p1)
t2 = time.time()
print("Time to complete numba prime function for number = {}: {} secs".format(p1,t2-t1))

t1 = time.time()
n_isp = is_prime_nb(p2)
t2 = time.time()
print("Time to complete numba prime function for number = {}: {} secs\n".format(p2,t2-t1))


## using Cython dedicated function

# Test times
t1 = time.time()
n_isp = is_prime_cython.is_prime_cy1(n)
t2 = time.time()
print("Time to complete cython prime function for number = {}: {} secs".format(n,t2-t1))

t1 = time.time()
n_isp = is_prime_cython.is_prime_cy1(p1)
t2 = time.time()
print("Time to complete cython prime function for number = {}: {} secs".format(p1,t2-t1))

t1 = time.time()
n_isp = is_prime_cython.is_prime_cy1(p2)
t2 = time.time()
print("Time to complete cython prime function for number = {}: {} secs\n".format(p2,t2-t1))

## using cython function with explicit data types
# Test times
t1 = time.time()
n_isp = is_prime_cython.is_prime_cy2(n)
t2 = time.time()
print("Time to complete cython prime function (with data types) for number = {}: {} secs".format(n,t2-t1))

t1 = time.time()
n_isp = is_prime_cython.is_prime_cy2(p1)
t2 = time.time()
print("Time to complete cython prime function (with data types) for number = {}: {} secs".format(p1,t2-t1))

t1 = time.time()
n_isp = is_prime_cython.is_prime_cy2(p2)
t2 = time.time()
print("Time to complete cython prime function (with data types) for number = {}: {} secs\n".format(p2,t2-t1))



###############################################
######## Benchmark 2: Multiprocessing #########

if __name__ =='__main__':
    # setup multiprocessing pool with 4 concurrent processes
    pool = mp.Pool(processes=4)
    
    # test time for multiple pools on basic pythonic function
    # note: the '10 * [p1]' input makes an array with len = 10 where p1 is repeated
    t1 = time.time()
    pool.map(is_prime, 10 * [p1])
    t2 = time.time()
    print("Time to complete multiprocessing (4 processes) on basic python function for number = {}: {} secs".format(p1,t2-t1))
    
    # numba function
    t1 = time.time()
    pool.map(is_prime_nb, 10 * [p2])
    t2 = time.time()
    print("Time to complete multiprocessing (4 processes) on numba function for number = {}: {} secs".format(p2,t2-t1))
    
    # cython function
    t1 = time.time()
    pool.map(is_prime_cython.is_prime_cy2, 10 * [p2])
    t2 = time.time()
    print("Time to complete multiprocessing (4 processes) on cython function (with data types) for number = {}: {} secs".format(p2,t2-t1))
    
    pool.close()


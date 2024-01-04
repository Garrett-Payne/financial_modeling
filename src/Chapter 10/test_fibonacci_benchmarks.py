## This script is intended to show procesing times for the Fibonacci sequence optimization
## Taken from chapter 10, pages 286 - 285 of Reference book

import time
import cython
import numba
from functools import lru_cache as cache
import pyximport; pyximport.install(pyimport=True,language_level=3)

# define standard function to run Fibonacci sequence
def fib_rec_py1(n): 
    if n<2:
        return n 
    else:
        return fib_rec_py1(n - 1) + fib_rec_py1(n - 2)

# Test times
t1 = time.time()
f1 = fib_rec_py1(35)
t2 = time.time()
print("Time to complete pythonic fibonacci function for number = {}: {} secs".format(35,t2-t1))

## NOTE: Numba has limited support for recursive functions, and I could not replicate what
# the book creates as a numba function for the reursive Fibonacci function. Since the numba
# speed up wasn't as fast as the cython implementation, I'm omitting it from this script.


# generate function using cython calls
def fib_rec_cy(n: cython.int):
    if n<2: 
        return n
    else:
        return fib_rec_cy(n - 1) + fib_rec_cy(n - 2)
    
# Test times
t1 = time.time()
f2 = fib_rec_cy(35)
t2 = time.time()
print("Time to complete cython fibonacci function (defined in this script) for number = {}: {} secs".format(35,t2-t1))

# test importing cython-specific function from .pyx file
import fibonacci_cython

# Test times
t1 = time.time()
f2 = fibonacci_cython.fib_rec_cy2(35)
t2 = time.time()
print("Time to complete cython fibonacci function (defined in separate pyx file) for number = {}: {} secs".format(35,t2-t1))


# test using cached intermediate results
@cache(maxsize=None) 
def fib_rec_py2(n):
    if n<2: return n
    else:
        return fib_rec_py2(n - 1) + fib_rec_py2(n - 2)
    
# Test times
t1 = time.time()
f2 = fib_rec_py2(35)
t2 = time.time()
print("Time to complete python fibonacci function (using cached memory) for number = {}: {} secs".format(35,t2-t1))


# use iterative method, rather than recursive
def fib_it_py(n): 
    x,y=0,1
    for i in range(1, n + 1): 
        x,y=y,x+y
    return x

# Test times
t1 = time.time()
f2 = fib_it_py(35)
t2 = time.time()
print("Time to complete pythonic fibonacci function (using iterative method) for number = {}: {} secs".format(35,t2-t1))


# convert iterative function into numba function
fib_it_nb = numba.jit(fib_it_py)

# Test times
t1 = time.time()
f2 = fib_it_nb(35)
t2 = time.time()
print("Time to complete numba fibonacci function (using iterative method) for number = {}: {} secs".format(35,t2-t1))

# test using cython function (iterative)
def fib_it_cy1(n: cython.int):
    i: cython.long
    x: cython.long = 0
    y: cython.long = 1
    for i in range(1, n + 1):
        x,y=y,x+y 
    return x

# Test times
t1 = time.time()
f2 = fib_it_cy1(35)
t2 = time.time()
print("Time to complete cython iterative fibonacci function (defined in this script) for number = {}: {} secs".format(35,t2-t1))

# test with imported cython iterative function

t1 = time.time()
f2 = fibonacci_cython.fib_it_cy2(35)
t2 = time.time()
print("Time to complete cython iterative fibonacci function (defined in separate pyx file) for number = {}: {} secs\n".format(35,t2-t1))


## Show how different functions can have different results - due to data type overflows
fn = fib_rec_py2(150)
print("Fibonacci sum (python iterative function) for n: {} = {}".format(150,fn))

fn1 = fib_it_nb(150)
print("Fibonacci sum (numba iterative function) for n: {} = {}".format(150,fn1))


fn2=fib_it_cy1(150)
print("Fibonacci sum (cython iterative function) for n: {} = {}".format(150,fn2))

print("\n Note that the numba method produces a different value, and depending on python version, cython could differ as well.\n")


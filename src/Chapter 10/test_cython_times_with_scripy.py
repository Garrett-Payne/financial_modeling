## This script is intended to show procesing times for various types of loops
## Taken from chapter 10, pages 280 - of Reference book
import random
import time
import cython

''' this below will allow us to load a cython .pyx file directly, 
and we won't have to run a setup.py script on the pyx file 
to directly compile it before importing into this script'''
import pyximport; pyximport.install(pyimport=True,language_level=3)
    
# define number of iterations to go through
n = 10000000 

def average_cy1(n: cython.int):
    i: cython.int 
    s: cython.float = 0 
    for i in range(n):
        s += random.random() 
    return s/n

t1 = time.time()
average_cy1(n)
t2 = time.time()
print("Time to complete average using cython defined function, # iterations = {}, {} secs\n".format(n,t2-t1))

print("Importing basic cython script. Should ouput 5 random numbers:")
## Import the cython script basic_cython_example.pyx
# Note that using pyximport allowed us to simply create the .pyx script & compile directly in this script:
# No setup.py script needed
import basic_cython_example
print("\n")

## Import function average_cy2 from basic_cython_funciton_example.pyx
# Note: didn't have to compile by using setup.py script either. pyximport handled it for us
print("Now importing basic cython function.")
import basic_cython_function_example

t1 = time.time()
cy2 =basic_cython_function_example.average_cy2(n)
t2 = time.time()
print("Time to complete average using cython defined function using stdlib library function, # iterations = {}, {} secs\n".format(n,t2-t1))


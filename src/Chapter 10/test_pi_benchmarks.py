## This script is intended to show procesing times for function to calculate digits of pi
## Taken from chapter 10, pages 290 - 293 of Reference book
import random 
import time
import numpy as np
import numba
from pylab import mpl, plt 
import cython
import pyximport; pyximport.install(pyimport=True,language_level=3)

# set some plotting/figure parameters
plt.style.use('seaborn-v0_8') 
mpl.rcParams['font.family'] = 'serif' 

# generate random number array
rn = [(random.random() * 2 - 1, random.random() * 2 - 1) for _ in range(500)]
rn = np.array(rn)

## Visualize the problem: Area of the circle is a fraction of a square (with same diameter).
## This fraction is equal to the amount of random points in circle divided by number of random points in square

# create new figure
fig = plt.figure(figsize=(7, 7))
# define axis object
ax = fig.add_subplot(1, 1, 1)

# create circle object
circ = plt.Circle((0, 0), radius=1, edgecolor='g', lw=2.0,
facecolor='None')
# create rectangle object
box = plt.Rectangle((-1, -1), 2, 2, edgecolor='b', alpha=0.3)
ax.add_patch(circ)  # add circle to figure
ax.add_patch(box)   # add rectangle to figure
# plot all the random values as red dots
plt.plot(rn[:, 0], rn[:, 1], 'r.') 
# set x and y axes limits
plt.ylim(-1.1, 1.1) 
plt.xlim(-1.1, 1.1)
plt.title("Visualization of Pi calculation")
plt.show(block=False)

# calculate random numbers using numpy
n = int(1e7)
t1 = time.time()
rn = np.random.random((n, 2)) * 2 - 1
t2 = time.time()
print("Time to create numpy array of {} random values: {} secs".format(n,t2-t1))

# # time how long it takes to calculate distance from center for each random point
t1 = time.time()
distance = np.sqrt((rn ** 2).sum(axis=1))
t2 = time.time()
print("Time to calculate magnitude for {} random values: {} secs".format(n,t2-t1))

# Time how long it takes to calculate fraction of points within circle compared to total points in square
t1 = time.time()
frac = (distance <= 1.0).sum() / len(distance)
t2 = time.time()
print("Time to calculate fraction of random points that lie within circle: {} secs".format(t2-t1))

# the value of pi is this fraction multiplied by the area of the square (4)
pi = frac * 4
print("Calculated value of pi: {}\n".format(pi))

## Define a function with for loop to iterate for pi
def mcs_pi_py(n): 
    circle = 0
    for _ in range(n):
        x, y = random.random(), random.random() 
        if(x**2+y**2)**0.5<=1:
            circle += 1 
    return (4 * circle) / n

# test time using basic function
t1 = time.time()
pi2 = mcs_pi_py(n)
t2 = time.time()
print("Time to calculate pi using iterative function with {} random values: {} secs".format(n,t2-t1))
print("Calculated value of pi: {}\n".format(pi2))

# generate function using numba
mcs_pi_nb = numba.jit(mcs_pi_py)

# test time using Numba function
t1 = time.time()
pi3 = mcs_pi_nb(n)
t2 = time.time()
print("Time to calculate pi using iterative Numba function with {} random values: {} secs".format(n,t2-t1))
print("Calculated value of pi: {}\n".format(pi3))

## Define an in-script cython function to calculate pi
def mcs_pi_cy1(n: cython.int):
    i: cython.int
    circle: cython.int = 0
    x: cython.float
    y: cython.float
    for i in range(n):
        x, y = random.random(), random.random() 
        if(x**2+y**2)**0.5<=1:
            circle += 1 
    return (4 * circle) / n

# test time using in-script cython function
t1 = time.time()
pi4 = mcs_pi_cy1(n)
t2 = time.time()
print("Time to calculate pi using iterative cython function with {} random values: {} secs".format(n,t2-t1))
print("Calculated value of pi: {}\n".format(pi4))

# test time using dedicated cython pyx script
import pi_cython

t1 = time.time()
pi4 = pi_cython.mcs_pi_cy1(n)
t2 = time.time()
print("Time to calculate pi using iterative cython (from pyx file) function with {} random values: {} secs".format(n,t2-t1))
print("Calculated value of pi: {}\n".format(pi4))

# test using cython pyx script with C library random function
t1 = time.time()
pi4 = pi_cython.mcs_pi_cy2(n)
t2 = time.time()
print("Time to calculate pi using iterative cython (from pyx file with C rand function) function with {} random values: {} secs".format(n,t2-t1))
print("Calculated value of pi: {}\n".format(pi4))
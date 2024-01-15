## This script is intended to show an example of symbolic computations
## Taken from chapter 11, pages 338 - 343 of Reference book
import numpy as np
import scipy.integrate as sci
from pylab import plt
from matplotlib.patches import Polygon
import sympy as sy

# create symbols
x = sy.Symbol('x')
y = sy.Symbol('y')
print("Type of x: {}:\n".format(type(x)))

#  define function f symbolically
f = x ** 2 + 3 + 0.5 * x ** 2 + 3 / 2
print("Simplified verison of f: \n{}\n".format(sy.simplify(f)))

# define sympy to use ASCII markups
sy.init_printing(pretty_print=False, use_unicode=False)
print("Pretty version of f: \n{}\n".format(sy.pretty(f)))

print("Another example of ASCII function mark-up: \n{}\n".format(sy.pretty(sy.sqrt(x) + 0.5)))

# use sympy to define 400,000 digits of pi
pi_str = str(sy.N(sy.pi, 400000))
print("first 40 digits of pi: {}\n".format(pi_str[:40]))

# find first index of a random 6-digit value in the first 400000 digits of pi
idx = pi_str.find('061072')
print("first index of number '061072' in pi: {}\n".format(idx))

## solving equations
# use sympy to solve different equations:
f = x ** 2 - 1
sln = sy.solve(f)
print("f = {} has solution: {}\n".format(f,sln))

f = x ** 2 - 1 - 3
sln = sy.solve(f)
print("f = {} has solution: {}\n".format(f,sln))

f = x ** 3 + 0.5 * x ** 2 - 1
sln = sy.solve(f)
print("f = {} has solution: {}\n".format(f,sln))

f = x ** 2 + y ** 2
sln = sy.solve(f)
print("f = {} has solution: {}\n".format(f,sln))


## integration and differentiation
#define some additional symbols to use as bounds
a, b = sy.symbols('a b')
I = sy.Integral(sy.sin(x) + 0.5 * x, (x, a, b))
print("Integral equation: \n {}\n".format(sy.pretty(I)))

# calculate the anti-derivative
int_func = sy.integrate(sy.sin(x) + 0.5 * x, x)
print("Anti-derivative: \n {}\n".format(sy.pretty(int_func)))

# evaluate anti-derivative values at bounds:
Fb = int_func.subs(x, 9.5).evalf()
Fa = int_func.subs(x, 0.5).evalf()
print("Integral value over bounds [0.5, 9.5]\n".format(Fb-Fa))

# calculate anti-derivate symbolically
int_func_limits = sy.integrate(sy.sin(x) + 0.5 * x, (x, a, b))
print("Anti-derivative with limits: \n {}\n".format(sy.pretty(int_func_limits)))

# solve the integral numerically with limits using dict substitution
int_val = int_func_limits.subs({a : 0.5, b : 9.5}).evalf()
print("Integral value, solved numerically (dict substitute): {}".format(int_val))

# solve integral numerically all in one line
sy.integrate(sy.sin(x) + 0.5 * x, (x, 0.5, 9.5))
print("Integral value, solved numerically (one line): {}\n".format(int_val))


## Derivatives
# reusing the function defined above:
# show the differentiation function
print("Derivative of above anti-derivative {}\n".format(int_func.diff()))


## define a new function to find derivative(s) for
f = (sy.sin(x) + 0.05 * x ** 2 + sy.sin(y) + 0.05 * y ** 2)
print("New function f = \n{}".format(sy.pretty(f)))

# calculate derivative with respect to x (partial derivative)
del_x = sy.diff(f, x)
print("Partial derivative of f with respect to x: {}".format(sy.pretty(del_x)))

# calculate derivative with respect to y (partial derivative)
del_y = sy.diff(f, y)
print("Partial derivative of f with respect to y: {}".format(sy.pretty(del_y)))

# Educated guesses for the roots and resulting optimal value
xo = sy.nsolve(del_x, -1.5)
print("Educated guesses of -1.5 for the root of df/dx and local min is found: {}".format(xo))

# Educated guesses for the roots and resulting optimal value
yo = sy.nsolve(del_y, -1.5)
print("Educated guesses of -1.5 for the root of df/dy and local min is found: {}".format(yo))

# use substitution function with local mins to find global min
glob_min = f.subs({x : xo, y : yo}).evalf()
print("global minimum of function: {}\n".format(glob_min))


# Be careful - bad guesses can lead to being trapped in local mins:
xo = sy.nsolve(del_x, 1.5)
print("Educated guesses of 1.5 for the root of df/dx and local min is found: {}".format(xo))
yo = sy.nsolve(del_y, 1.5)
print("Educated guesses of -1.5 for the root of df/dy and local min is found: {}".format(yo))
print("Incorrect guesses leads to finding local min, not global min. Min found with the above inputs as: {}\n".format(f.subs({x : xo, y : yo}).evalf()))
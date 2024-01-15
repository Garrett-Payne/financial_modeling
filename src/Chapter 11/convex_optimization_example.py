## This script is intended to show an example of convex optimization
## Taken from chapter 11, pages 329 - 334 of Reference book
import numpy as np
from pylab import plt, mpl
from mpl_toolkits.mplot3d import Axes3D #used to make 3D plots
import scipy.optimize as sco
import math

# plotting parameters
plt.style.use('seaborn-v0_8')
mpl.rcParams['font.family'] = 'serif'


# define a 3D function as an example
def fm(p):
    x, y = p
    return (np.sin(x) + 0.05 * x ** 2+ np.sin(y) + 0.05 * y ** 2)

# create x and y data arrays, meshgrids, and then the dependent variable Z
x = np.linspace(-10, 10, 50)
y = np.linspace(-10, 10, 50)
X, Y = np.meshgrid(x, y)
Z = fm((X, Y))

# plot the data
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(projection='3d')
surf = ax.plot_surface(X, Y, Z, rstride=2, cstride=2,
cmap='coolwarm', linewidth=0.5,
antialiased=True)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x, y)')
fig.colorbar(surf, shrink=0.5, aspect=5,ax=ax)
plt.title("3D Function Example")
plt.show()

output = True

# create a function to optionally output to terminal
def fo(p):
    x, y = p
    z = np.sin(x) + 0.05 * x ** 2 + np.sin(y) + 0.05 * y ** 2
    if output == True:
        print('%8.4f | %8.4f | %8.4f' % (x, y, z))
    return z

# Try the brute force minimizer

print("Running the brute force minimizer")
optimal = sco.brute(fo, ((-10, 10.1, 5), (-10, 10.1, 5)), finish=None)
print("The global minimum is found by the brute force optimizer as: {}".format(optimal))

output=False
opt1 = sco.brute(fo, ((-10, 10.1, 0.1), (-10, 10.1, 0.1)), finish=None)
print("The global minimum is found by the brute force optimizer (with smaller step size) as: {}\n".format(opt1))


## Local optimization
output = True

# this function will take in the same function, along with an initial starting paramater, to fine tune the 
# local minimum
opt2 = sco.fmin(fo, opt1, xtol=0.001, ftol=0.001,maxiter=15, maxfun=20)


# however, if given different initial paramaters, the local minimum may be found at another point:
output = False
optmin = sco.fmin(fo, (2.0, 2.0), maxiter=250)
print("The local minimum , starting at point [2,2], is found by the fmin function as: {}\n".format(optmin))


## So far, these methods have been shown how to find the optimal (or sub-optimal) solution for unconstrained problems.
# we'll next look at constrained functions and optimization

# we'll define the Expected Utility Modeling function
# which is defined as
# max(a,b) E(u(w_1)) = p * sqrt(w_1u) + (1-p) * sqrt(w_1d)
# w_1u = a * r_a + b * r_b
# w_0 >= a * q_a + b * q_b
# a,b > 0

# Define constants as:
# q_a = q_b = 10 are security costs
# r_a = [15, 5] payoff for state u
# r_b = [5, 12] payoff for state d
# w0 = 100 investor budget
# p = .5 (equally likely for u or d)
# utitility function: u(w) = sqrt(w), where w is wealth

# this leads us to the objective: minimize the function
# min(a,b) - E(u(w1)) = -(.5*sqrt(w_1u) + 0.5*sqrt(w_1d))
# with definitions:
# w1_u = a*15 + b*5
# w1_d = a*5 + b*12
# and constrant:
# 100 >= a*10 + b*10

# define the function to be minimized - the expected utility function
# note that this matches the definition above, except variable s = a in book notation
def Eu(p):
    s, b = p
    return -(0.5 * math.sqrt(s * 15 + b * 5) +0.5 * math.sqrt(s * 5 + b * 12))

# now we'll define the constraint as an inequality function
# and use a lambda function to define the constraint
cons = ({'type': 'ineq','fun': lambda p: 100 - p[0] * 10 - p[1] * 10})

# define the boundaries to search for a & b
bnds = ((0, 1000), (0, 1000))

# run the minimize function on the given function, initial guess as [5,5], 
# using the Sequential Least Squares Programming (SLSQP), given bounds & constraints
result = sco.minimize(Eu, [5, 5], method='SLSQP',bounds=bnds, constraints=cons)

print("The optimal portfolio is found as: {}".format(result['x']))
print("The optimal solution provides a payout of {}".format(-result['fun']))
print("The constraint binding: {}\n".format(np.dot(result['x'], [10, 10])))


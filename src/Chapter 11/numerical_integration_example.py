## This script is intended to show an example of numerical integration
## Taken from chapter 11, pages 335 - 337 of Reference book
import numpy as np
import scipy.integrate as sci
from pylab import plt
from matplotlib.patches import Polygon

# define function to integrate
def f(x):
    return np.sin(x) + 0.5 * x

# create array of points
x = np.linspace(0, 10)

# define dependent values from function
y = f(x)

# define lower and upper integration bounds
a = 0.5
b = 9.5

# define array of integration bounds
Ix = np.linspace(a, b)
# and function points 
Iy = f(Ix)

## plot the function - the integral value is the area under the blue line (grey area)
fig, ax = plt.subplots(figsize=(10, 6))
plt.plot(x, y, 'b', linewidth=2)
plt.ylim(bottom=0)
Ix = np.linspace(a, b)
Iy = f(Ix)
verts = [(a, 0)] + list(zip(Ix, Iy)) + [(b, 0)]
poly = Polygon(verts, facecolor='0.7', edgecolor='0.5')
ax.add_patch(poly)
plt.text(0.75 * (a + b), 1.5, r"$\int_a^b f(x)dx$",horizontalalignment='center', fontsize=20)
plt.figtext(0.9, 0.075, '$x$')
plt.figtext(0.075, 0.9, '$f(x)$')
ax.set_xticks((a, b))
ax.set_xticklabels(('$a$', '$b$'))
ax.set_yticks([f(a), f(b)])
plt.show()

# test 3 different methods for numerical integration
# note: all of these methods take the function and the upper & lower bounds as inputs
integral = sci.fixed_quad(f, a, b)[0]
print("The numerical integral, using fixed Gaussian quadrature: {}".format(integral))

integral = sci.quad(f, a, b)[0]
print("The numerical integral, using adaptive quadrature: {}".format(integral))

integral = sci.romberg(f, a, b)
print("The numerical integral, using Romberg integration: {}".format(integral))

# some other numerical techniques take the full array of data to integrate over
xi = np.linspace(0.5, 9.5, 25)

integral = sci.trapz(f(xi), xi)
print("The numerical integral, using trapezoidal integration: {}".format(integral))

integral = sci.simps(f(xi), xi)
print("The numerical integral, using Simpson's rule: {}\n".format(integral))

## integration by simulation
# the below code implements a basic Monte Carlo simulation to calculate the integral
print("Monte Carlo results, increasing number of random points by 10 for each iteration:")
for i in range(1, 20):
    np.random.seed(1000)
    x = np.random.random(i * 10) * (b - a) + a
    print(np.mean(f(x)) * (b - a))
print("\n")


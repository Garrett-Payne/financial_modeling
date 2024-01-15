## This script is intended to show an example of numerial approximation
## Taken from chapter 11, pages 312 - 328 of Reference book
import numpy as np
from pylab import plt, mpl
from mpl_toolkits.mplot3d import Axes3D #used to make 3D plots
import scipy.interpolate as spi

plt.style.use('seaborn-v0_8')
mpl.rcParams['font.family'] = 'serif'

# define a basic function - combines trig and linear addition
def f(x):
    return np.sin(x) + 0.5 * x

# create a function to use for plotting
def create_plot(x, y, styles, labels, axlabels):
    plt.figure(figsize=(10, 6))
    for i in range(len(x)):
        plt.plot(x[i], y[i], styles[i], label=labels[i])
        plt.xlabel(axlabels[0])
        plt.ylabel(axlabels[1])
    plt.legend(loc=0)


# create array of 50 evenly spaced points to use as independent variables
x = np.linspace(-2 * np.pi, 2 * np.pi, 50)

# and plot the data to be approximated
create_plot([x], [f(x)], ['b'], ['f(x)'], ['x', 'f(x)'])
plt.title("Data to be Approximated")
plt.show(block=False)

##### Regression -
# example shows how to run a regression model. This example uses monomials to approximate the function
# aka using an equation in the form of a + bx + cx**2 + dx**3 + ... with Nth order
# this example uses numpy.polyfit() function to fit to a curve and calculate the polynomial coefficients,
# then use np.polyeval() to apply the coefficients to some x independent variables

# calculate 1st power polynomial (aka linear regression - just a line)
res = np.polyfit(x, f(x), deg=1, full=True)

# evaluate polynomial - aka calculate dependent variable points for each independent point (x)
ry = np.polyval(res[0], x)

# and let's plot it
create_plot([x, x], [f(x), ry], ['b', 'r.'],['f(x)', 'regression'], ['x', 'f(x)'])
plt.title("Linear Regression Model")
plt.show(block=False)

# now try out using higher-order polynomial
reg = np.polyfit(x, f(x), deg=5)
ry = np.polyval(reg, x)
create_plot([x, x], [f(x), ry], ['b', 'r.'],['f(x)', 'regression'], ['x', 'f(x)'])
plt.title("5th-Order Polynomial Approximation")
plt.show(block=False)

# calculate 7th-order approximation
reg = np.polyfit(x, f(x), 7)
ry = np.polyval(reg, x)
# calculate the Mean Square Error (MSE) of the approximation
MSE = np.mean((f(x) - ry) ** 2)
print("Mean Square Error (MSE) of 7th-Order Polynomial Regression Model: {}\n".format(MSE))
create_plot([x, x], [f(x), ry], ['b', 'r.'],['f(x)', 'regression'], ['x', 'f(x)'])
plt.title("7th-Order Polynomial Approximation")
plt.show(block=False)

## let's test using different basis functions for the regression

# first, create a numpy array for 3rd order monomials
matrix = np.zeros((3 + 1, len(x)))
matrix[3, :] = x ** 3
matrix[2, :] = x ** 2
matrix[1, :] = x
matrix[0, :] = 1

# now calculate regression using least squares function
# this function calculates the least squares solution to an input set of linear equations 
# (aka the input matrix)
reg = np.linalg.lstsq(matrix.T, f(x), rcond=None)[0]

# calculate the dependent variables from the reg matrix
ry = np.dot(reg, matrix)
create_plot([x, x], [f(x), ry], ['b', 'r.'],['f(x)', 'regression'], ['x', 'f(x)'])
plt.title("Least Squares Approximation, 3rd Order, Using Monomials")
plt.show(block=False)

## Now try using sin(x) as one of the basis functions
matrix[3, :] = np.sin(x)
reg = np.linalg.lstsq(matrix.T, f(x), rcond=None)[0]
ry = np.dot(reg, matrix)
create_plot([x, x], [f(x), ry], ['b', 'r.'],['f(x)', 'regression'], ['x', 'f(x)'])
plt.title("Least Squares Approximation, 3rd Order + sin(x)")
plt.show(block=False)
MSE = np.mean((f(x) - ry) ** 2)
print("Mean Square Error (MSE) of 3rd Order Polynomial + Sin(x) Least Squares Approximation: {}\n".format(MSE))


### So far we've worked with clean data - an exact function. Let's test approximation using noisy data

# generate array of 50 evenly-spaced values
xn = np.linspace(-2 * np.pi, 2 * np.pi, 50)
# add noise as random values sampled from a sample deviation to the independent variables
xn = xn + 0.15 * np.random.standard_normal(len(xn))
# now run the data through the function to calcualate the dependent variables & then
# add random noise
yn = f(xn) + 0.25 * np.random.standard_normal(len(xn))

# calculate 7th-Order polynomial fit to new noisy dataset
reg = np.polyfit(xn, yn, 7)
ry = np.polyval(reg, xn)
create_plot([x, x], [f(x), ry], ['b', 'r.'],['f(x)', 'regression'], ['x', 'f(x)'])
plt.title("7th-Order Polynomial Approximation of Noisy Data")
plt.show(block = False)


## Now let's test with unsorted data-

# instead of making an array of evenly-spaced, always increasing values, we'll create
# an array of randomly selected values to use as independent variable
xu = np.random.rand(50) * 4 * np.pi - 2 * np.pi
# calculate dependent variable using defined function
yu = f(xu)

# calculate 5th-Order polynomial for unsorted data
reg = np.polyfit(xu, yu, 5)
ry = np.polyval(reg, xu)
create_plot([xu, xu], [yu, ry], ['b.', 'ro'],['f(x)', 'regression'], ['x', 'f(x)'])
plt.title("5th-Order Polynomial Approximation, Unsorted Data")
plt.show(block=False)

## let's now try approximation with multiple (>1) dimensions

# define new function that takes in tuple with 2 independent variables and calculates a 3rd, dependent variable
def fm(p):
    x, y = p
    return np.sin(x) + 0.25 * x + np.sqrt(y) + 0.05 * y ** 2

# create arrays for 2 independent variables
x = np.linspace(0, 10, 20)
y = np.linspace(0, 10, 20)
# now create uniform meshgrid objects
X, Y = np.meshgrid(x, y)
# run data through function to get 3rd dimension
Z = fm((X, Y))
# and flatten the data from matrix form to vector
x = X.flatten()
y = Y.flatten()

# now let's plot the 3D data
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(projection='3d')
surf = ax.plot_surface(X, Y, Z, rstride=2, cstride=2,
cmap='coolwarm', linewidth=0.5,
antialiased=True)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x, y)')
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.title("3D Function Example")
plt.show(block=False)

# now create matrix for approximation, with some knowledge of how the actual function was created
# so that we can define the correct basis functions
matrix = np.zeros((len(x), 6 + 1))
matrix[:, 6] = np.sqrt(y)
matrix[:, 5] = np.sin(x)
matrix[:, 4] = y ** 2
matrix[:, 3] = x ** 2
matrix[:, 2] = y
matrix[:, 1] = x
matrix[:, 0] = 1

# calculate least squares estimate using input matrix
reg = np.linalg.lstsq(matrix, fm((x, y)), rcond=None)[0]
RZ = np.dot(matrix, reg).reshape((20, 20))

# and let's plot the approximation
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(projection='3d')
surf1 = ax.plot_surface(X, Y, Z, rstride=2, cstride=2,
cmap=mpl.cm.coolwarm, linewidth=0.5,
antialiased=True)
surf2 = ax.plot_wireframe(X, Y, RZ, rstride=2, cstride=2,
label='regression')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x, y)')
ax.legend()
# note: the ax=ax below makes sure the colorbar shows up on this plot. It was showing up on the previous plot
# for me until I added this input
fig.colorbar(surf, shrink=0.5, aspect=5,ax=ax)
plt.title("3D Function LSE Approximation")
plt.show(block=False)



##### INTERPOLATION #####
# let's test out interpolation - which is trying to approximate data points in-between observed points

# create array of data
x = np.linspace(-2 * np.pi, 2 * np.pi, 25)

#define new function to run data through
def f(x):
    return np.sin(x) + 0.5 * x

# create spline interpolation, with 1st order polynomial (k=1) - linear spline
ipo = spi.splrep(x, f(x), k=1)

# create the interpolated data based on inputs
iy = spi.splev(x, ipo)

# and plot it
create_plot([x, x], [f(x), iy], ['b', 'ro'],['f(x)', 'interpolation'], ['x', 'f(x)'])
plt.title("1st-Order Interpolation example")
plt.show(block=False)

# create different array of data for same function
xd = np.linspace(1.0, 3.0, 50)

# and run through the linear spline model to generate new interpolation points
iyd = spi.splev(xd, ipo)
create_plot([xd, xd], [f(xd), iyd], ['b', 'ro'],['f(x)', 'interpolation'], ['x', 'f(x)'])
plt.title("1st-Order Interpolation Example, Different Interpolation Points")
plt.show(block=False)

# let's redo spline, this time make it 3rd-order (cubic spline)
ipo = spi.splrep(x, f(x), k=3)
iyd = spi.splev(xd, ipo)
MSE = np.mean((f(xd) - iyd) ** 2)
print("Mean Square Error (MSE) of Cubic Spline Interpolation: {}\n".format(MSE))

create_plot([xd, xd], [f(xd), iyd], ['b', 'ro'],['f(x)', 'interpolation'], ['x', 'f(x)'])
plt.title("3rd-Order Interpolation Example, Different Interpolation Points")
plt.show()




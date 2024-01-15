## This script is intended to show examples of different simulations
## Taken from chapter 12, pages 353 - 372 of Reference book
import math
import numpy as np
import numpy.random as npr
from pylab import plt, mpl
import scipy.stats as scs
import time

plt.style.use('seaborn-v0_8')
mpl.rcParams['font.family'] = 'serif'

# example function to model: the Black-Scholes-Merton setup
# follows equation:
# S_T = S_0*exp[ (r - .5*sigma**2)*T + sigma*sqrt(T)*z]
# where:
# S_T = Index level at date T
# r = Constant riskless short rate
# sigma = constant volatility (standard deviation of returns) of S
# z = standard normally distributed random variable

# set initial values:
S0 = 100
r = 0.05
sigma = 0.25
T = 2.0 # timeline (in years)
I = 10000 # number of iterations/ random values

# model Monte Carlo using Numpy random numbers - uses I amount of random numbers for evaluation
ST1 = S0 * np.exp((r - 0.5 * sigma ** 2) * T +sigma * math.sqrt(T) * npr.standard_normal(I))

plt.figure(figsize=(10, 6))
plt.hist(ST1, bins=50)
plt.xlabel('index level')
plt.ylabel('frequency')
plt.title("Monte Carlo Results of the Black-Scholes-Merton setup")
plt.show(block = False)

# retry Monte Carlo using log normal random values
# first value is mean, second is standard deviation
ST2 = S0 * npr.lognormal((r - 0.5 * sigma ** 2) * T,sigma * math.sqrt(T), size=I)

plt.figure(figsize=(10, 6))
plt.hist(ST2, bins=50)
plt.xlabel('index level')
plt.ylabel('frequency')
plt.title("Monte Carlo Results of the Black-Scholes-Merton setup, using npr.lognormal distribution")
plt.show(block = False)

# both distributions look similar, but we can use some statistical tests to compare the two distributions

# first, create a standardized function to show what to output to console from comparing two distributions
def print_statistics(a1, a2):
    ''' Prints selected statistics.
    Parameters ==========
    a1, a2: ndarray objects
    results objects from simulation
    '''
    sta1 = scs.describe(a1)
    sta2 = scs.describe(a2)
    print('%14s %14s %14s' %
    ('statistic', 'data set 1', 'data set 2'))
    print(45 * "-")
    print('%14s %14.3f %14.3f' % ('size', sta1[0], sta2[0]))
    print('%14s %14.3f %14.3f' % ('min', sta1[1][0], sta2[1][0]))
    print('%14s %14.3f %14.3f' % ('max', sta1[1][1], sta2[1][1]))
    print('%14s %14.3f %14.3f' % ('mean', sta1[2], sta2[2]))
    print('%14s %14.3f %14.3f' % ('std', np.sqrt(sta1[3]),np.sqrt(sta2[3])))
    print('%14s %14.3f %14.3f' % ('skew', sta1[4], sta2[4]))
    print('%14s %14.3f %14.3f\n' % ('kurtosis', sta1[5], sta2[5]))

print("Compare models:")
print_statistics(ST1, ST2)

## Let's apply Geometric Brownian Motion 
# consider the Black-Scholes-Merton model in its dynamic form, described by a stochastic differential equation (SDE):
# dS_t = r * S_t * dt + sigma * S_t * dZ_t
# where:
# values of Stare log-normally distributed
# applying an Euler scheme, we can get the equation discretized:
# S_t = S_t-delta_t * exp[(r - .5*sigma**2)*delta_t + sigma*sqrt(delta_t)*z_t]

# setup model paramamters
I = 10000 # number of iterations
M = 50 # number of time intervals
dt = T / M # length of each time interval

# setup model - can initialize numpy array for speed
S = np.zeros((M + 1, I))
S[0] = S0 # set initial value

# run model in for loop
for t in range(1, M + 1):
    S[t] = S[t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt +sigma * math.sqrt(dt) * npr.standard_normal(I))

# and plot the data
plt.figure(figsize=(10, 6))
plt.hist(S[-1], bins=50)
plt.xlabel('index level')
plt.ylabel('frequency')
plt.title("Black-Scholes-Merton Model Dynamic Simulation")
plt.show(block=False)

print("compare dynamic model with static model:")
print_statistics(S[-1], ST2)

# plot first 10 paths to show Brownian motion
plt.figure(figsize=(10, 6))
plt.plot(S[:, :10], lw=1.5)
plt.xlabel('time')
plt.ylabel('index level')
plt.title("Dynamically simulated geometric Brownian motion paths")
plt.show(block = False)


## let's try a different model: mean-reverting processes
# setup example as square root diffusion -proposed by Cox, Ingersoll, and Ross
# This follows SDE:
# dx_t = k * (theta - x_t) * dt + sigma * sqrt(x_t) * dZ_t
# where:
# x_t is process level at date t
# k is mean-reversion factor (technically variable is kappa)
# theta is long-term mean of the process
# sigma is constant volatility parameter
# Z_t is standard Brownian motion
#
# This SDE can also be discretized using Euler Scheme:
# x_h_t = x_h_s + k * (theta - x_h_s_plus) * delta_t + sigma * sqrt(x_h_s_plus)*sqrt(delta_t) * z_t
# x_t = x_h_s_plus
# where
# s = t - delta_t
# x_plus = max(x, 0)

# define parameters:
x0 = 0.05
kappa = 3.0
theta = 0.02
sigma = 0.1
I = 10000
M = 50
dt = T / M

# define model using function
def srd_euler():
    xh = np.zeros((M + 1, I))
    x = np.zeros_like(xh)
    xh[0] = x0
    x[0] = x0
    for t in range(1, M + 1):
        xh[t] = (xh[t - 1] + kappa * (theta - np.maximum(xh[t - 1], 0)) * dt +
        sigma * np.sqrt(np.maximum(xh[t - 1], 0)) *
        math.sqrt(dt) * npr.standard_normal(I))
        x = np.maximum(xh, 0)
    return x

# run model
x1 = srd_euler()

# and plot 
plt.figure(figsize=(10, 6))
plt.hist(x1[-1], bins=50)
plt.xlabel('value')
plt.ylabel('frequency')
plt.title("square-root diffusion model")
plt.show(block = False)

# plot first 10 paths to show Brownian motion
plt.figure(figsize=(10, 6))
plt.plot(x1[:, :10], lw=1.5)
plt.xlabel('time')
plt.ylabel('index level')
plt.title("Dynamically simulated square-root diffusion paths (Euler scheme)")
plt.show(block=False)

# discretizing the model, based on the noncentral chi-squared distribution chi**2
# with:
# df = (4 * theta * k)/sigma**2
# (df = degrees of freedom)
#
# nc = [4 * k * exp(-k*delta_t) ]/ [sigma**2 * (1 - exp(-k*delta_t) )] * x_s
# (nc = noncentrality parameter)
#
# These lead to the exact discretized equation:
# x_t = sigma**2* [1 - exp(-k*delta_t)]/4*k * chi_d**2 [4 * k * exp(-k*delta_t) ]/ [sigma**2 * (1 -exp(-k*delta_t) )] * x_s]
# this is implemented in function below

def srd_exact():
    x = np.zeros((M + 1, I))
    x[0] = x0
    for t in range(1, M + 1):
        df = 4 * theta * kappa / sigma ** 2
        c = (sigma ** 2 * (1 - np.exp(-kappa * dt))) / (4 * kappa)
        nc = np.exp(-kappa * dt) / c * x[t - 1]
        x[t] = c * npr.noncentral_chisquare(df, nc, size=I)
    return x

# run model
x2 = srd_exact()

# plot results
plt.figure(figsize=(10, 6))
plt.hist(x2[-1], bins=50)
plt.xlabel('value')
plt.ylabel('frequency')
plt.title("Disretization of square-root diffusion, using Chi-Squared Distribution")
plt.show(block=False)

# show first 10 paths of random 
plt.figure(figsize=(10, 6))
plt.plot(x2[:, :10], lw=1.5)
plt.xlabel('time')
plt.ylabel('index level')
plt.title("Dynamically simulated square-root diffusion paths (Chi-Squared Distribution Discretization)")
plt.show(block=False)

print("Comparing Euler Scheme vs. Chi-Squared for square-root diffusion discretization:")
print_statistics(x1[-1], x2[-1])

# compare time to compute each model:

# increase number of iterations
I = 250000

t1= time.time()
x1 = srd_euler()
t2 = time.time()
print("Time to run Euler discretization scheme on {} iterations: {} secs".format(I,t2-t1))

t1 = time.time()
x2 = srd_exact()
t2 = time.time()
print("Time to run Chi-Squared discretization scheme on {} iterations: {} secs\n".format(I,t2-t1))

print("Comparing Euler discretization vs. Chi-Squared discretization on {} iterations:".format(I))
print_statistics(x1[-1], x2[-1])


## The Black-Scholes-Merton model above has assumed volatility is constant
# what if we modeled it as a stochastic process instead - we'll introduce stochastic differential equations (SDEs)
# for stochastic volatility models
# the equations to follow are Stochastic differential equations for Heston stochastic volatility,
# given by:
# dS_t = r * S_t * dt + sqrt(v_t) * S_t * dZ_t_1
# dv_t = k_v * (theta_v - v_t) * dt + sigma_v * sqrt(v_t) * dZ_t_2
# dZ_t_1 * dZ_t_2 = rho
# where
# rho is instantaneous correlation between the two standard Brownian motions Z_t_1 & Z_t_2

# setup parameters for new model
S0 = 100.
r = 0.05
v0 = 0.1 # initial volatility
kappa = 3.0
theta = 0.25
sigma = 0.1
rho = 0.6 # fixed correlation value between 2 Brownian motion models
T = 1.0

# Use Cholesky decomposition to calculate correlation between two different stochastic processes
corr_mat = np.zeros((2, 2))
corr_mat[0, :] = [1.0, rho]
corr_mat[1, :] = [rho, 1.0]
cho_mat = np.linalg.cholesky(corr_mat)

# now generate model following Euler Scheme:
M = 50
I = 10000
dt = T / M

# simulate Brownian motion using above SDE

# generate numpy random matrix 
ran_num = npr.standard_normal((2, M + 1, I))
v = np.zeros_like(ran_num[0])
vh = np.zeros_like(v)
v[0] = v0
vh[0] = v0
for t in range(1, M + 1):
    ran = np.dot(cho_mat, ran_num[:, t, :])
    vh[t] = (vh[t - 1] +
        kappa * (theta - np.maximum(vh[t - 1], 0)) * dt +
        sigma * np.sqrt(np.maximum(vh[t - 1], 0)) *
        math.sqrt(dt) * ran[1])    
v = np.maximum(vh, 0)

# now calculate model using dynamic v values
S = np.zeros_like(ran_num[0])
S[0] = S0
for t in range(1, M + 1):
    ran = np.dot(cho_mat, ran_num[:, t, :])
    S[t] = S[t - 1] * np.exp((r - 0.5 * v[t]) * dt +
        np.sqrt(v[t]) * ran[0] * np.sqrt(dt))
    
# plot the data
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
ax1.hist(S[-1], bins=50)
ax1.set_xlabel('index level')
ax1.set_ylabel('frequency')
ax2.hist(v[-1], bins=50)
ax2.set_xlabel('volatility')
ax1.set_title("Dynamically simulated stochastic volatility process at maturity")
plt.show(block=False)

# compare maturity model to volatility:
print("Compare model to stochastic volatility:")
print_statistics(S[-1], v[-1])

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True,
figsize=(10, 6))
ax1.plot(S[:, :10], lw=1.5)
ax1.set_ylabel('index level')
ax2.plot(v[:, :10], lw=1.5)
ax2.set_xlabel('time')
ax2.set_ylabel('volatility')
ax1.set_title("Dynamically simulated stochastic volatility process paths")
plt.show(block=False)

## analzying jump diffusion
# we'll use the Stochastic differential equation for Merton jump diffusion model:
# dS_t = (r - r_j) * S_t * dt + sigma * S_t * dZ_t + J_t * S_t * dN_t
# where:
# S_t = index level at date t
# r = constant riskless short rate
# r_j = lambda * (exp(mu_j + delta**2/2) -1)
# which is drift correction for jump to maintain risk neutrality
# sigma = constant volatility of S
# Z_t = standard Brownian motion
# J_t = Jump at date t, with distribution:
# log(1 + J_t) ~= N( log(1 + mu_j) - delta**2/2, delta**2)
# N is the cumulative distribution function of a standard normal variable
# N_t is a Poisson process with intesity lambda
#
# Applying Euler discretization:
# S_t = S_t_minus_Delta_t * [exp( (r - r_j - sigma**2/2)*Delta_t * sigma*sqrt(Delta_t)*z_t_1 ) + (exp(mu_j + delta*z_t_2) - 1)*y_t]

# setup model paramaters
S0 = 100.
r = 0.05
sigma = 0.2
lamb = 0.75 # jump intensity
mu = -0.6 # mean jump size
delta = 0.25 # jump volatility
rj = lamb * (math.exp(mu + 0.5 * delta ** 2) - 1) # drift correction
T = 1.0
M = 50
I = 10000
dt = T / M

S = np.zeros((M + 1, I))
S[0] = S0
sn1 = npr.standard_normal((M + 1, I))
sn2 = npr.standard_normal((M + 1, I))
poi = npr.poisson(lamb * dt, (M + 1, I))
for t in range(1, M + 1, 1):
    S[t] = S[t - 1] * (np.exp((r - rj - 0.5 * sigma ** 2) * dt +
        sigma * math.sqrt(dt) * sn1[t]) +
        (np.exp(mu + delta * sn2[t]) - 1) *
        poi[t])
    S[t] = np.maximum(S[t], 0)

# plot results
plt.figure(figsize=(10, 6))
plt.hist(S[-1], bins=50)
plt.xlabel('value')
plt.ylabel('frequency')
ax1.set_title("Dynamically simulated jump diffusion process at maturity")
plt.show(block=False)

# plot first 10 paths
plt.figure(figsize=(10, 6))
plt.plot(S[:, :10], lw=1.5)
plt.xlabel('time')
plt.ylabel('index level')
plt.title("Dynamically simulated jump diffusion process paths")
plt.show()
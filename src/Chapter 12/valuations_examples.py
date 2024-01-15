## This script is intended to show examples of valuations using MCS
## Taken from chapter 12, pages 375 - 383 of Reference book
import math
import numpy as np
import numpy.random as npr
from pylab import plt, mpl
import os
import sys

plt.style.use('seaborn-v0_8')
mpl.rcParams['font.family'] = 'serif'

# re-defined from variance_reduction.py script
def gen_sn(M, I, anti_paths=True, mo_match=True):
    ''' Function to generate random numbers for simulation.
    Parameters
    ==========
    M: int
    number of time intervals for discretization
    I: int
    number of paths to be simulated
    anti_paths: boolean
    use of antithetic variates
    mo_math: boolean
    use of moment matching
    '''
    if anti_paths is True:
        sn = npr.standard_normal((M + 1, int(I / 2)))
        sn = np.concatenate((sn, -sn), axis=1)
    else:
        sn = npr.standard_normal((M + 1, I))
    if mo_match is True:
        sn = (sn - sn.mean()) / sn.std()
    return sn

# the first example looks at the European option - 
# the Pricing by risk-neutral expectation is given by the following equation:
# C_0 = exp(-r*T) * Integral /0 to infinity [h(s) * q(s) * ds]
# where h(t) = max(S_t - K, 0)is payoff on European call option
# S_t is index level at maturity date t
# K is strike price
# This integral can be evaluated with Monte Carlo Sim:
# The risk-neutral Monte Carlo estimator is given as
# C_0_tilde = exp(-r * T) * (1/L) * Sum (from i = 1::=L) h(S_T_i_tilde)
#
# This can be modeled by:

# first, define static parameters
S0 = 100.
r = 0.05
sigma = 0.25
T = 1.0
I = 50000

# now define C0 as a function of K
def gbm_mcs_stat(K):
    ''' Valuation of European call option in Black-Scholes-Merton
    by Monte Carlo simulation (of index level at maturity)
    Parameters
    ==========
    K: float
    (positive) strike price of the option
    Returns
    =======
    C0: float
    estimated present value of European call option
    '''
    sn = gen_sn(1, I)
    # simulate index level at maturity
    ST = S0 * np.exp((r - 0.5 * sigma ** 2) * T
        + sigma * math.sqrt(T) * sn[1])
    # calculate payoff at maturity
    hT = np.maximum(ST - K, 0)
    # calculate MCS estimator
    C0 = math.exp(-r * T) * np.mean(hT)
    return C0

print("The Monte Carlo estimator value for the European call option: {}\n".format(gbm_mcs_stat(K=105.)))

## Now calculate dynamic simulation approach - allows for European put option along with call option
M = 50
def gbm_mcs_dyna(K, option='call'):
    ''' Valuation of European options in Black-Scholes-Merton
    by Monte Carlo simulation (of index level paths)
    Parameters
    ==========
    K: float
    (positive) strike price of the option
    option : string
    type of the option to be valued ('call', 'put')
    Returns
    =======
    C0: float
    estimated present value of European call option
    '''
    dt = T / M
    # simulation of index level paths
    S = np.zeros((M + 1, I))
    S[0] = S0
    sn = gen_sn(M, I)
    for t in range(1, M + 1):
        S[t] = S[t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt
        + sigma * math.sqrt(dt) * sn[t])
    # case-based calculation of payoff
    if option == 'call':
        hT = np.maximum(S[-1] - K, 0)
    else:
        hT = np.maximum(K - S[-1], 0)
    # calculation of MCS estimator
    C0 = math.exp(-r * T) * np.mean(hT)
    return C0


# run the sim for call and for put option
print("The Dynamic Monte Carlo estimator value for the European call option: {}\n".format(gbm_mcs_dyna(K=110., option='call')))

print("The Dynamic Monte Carlo estimator value for the European put option: {}\n".format(gbm_mcs_dyna(K=110., option='put')))


## compare to Black-Scholes-Merton-formulation
# First, we compare the results from the static simulation approach with precise analytical values:

# use the pre-defined bsm_functions.py script for exact solution
current = os.path.dirname(os.path.realpath(__name__))
parent = os.path.dirname(current)
sys.path.append(parent)
from bsm_functions import bsm_call_value
stat_res = []
dyna_res = []
anal_res = []
# Creates an ndarray object containing the range of strike prices
k_list = np.arange(80., 120.1, 5.)
np.random.seed(100)
#Simulates/calculates and collects the option values for all strike prices.
for K in k_list:
    stat_res.append(gbm_mcs_stat(K))
    dyna_res.append(gbm_mcs_dyna(K))
    anal_res.append(bsm_call_value(S0, K, T, r, sigma))
stat_res = np.array(stat_res)
dyna_res = np.array(dyna_res)
anal_res = np.array(anal_res)

# plot comparison - for static model
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
ax1.plot(k_list, anal_res, 'b', label='analytical')
ax1.plot(k_list, stat_res, 'ro', label='static')
ax1.set_ylabel('European call option value')
ax1.legend(loc=0)
ax1.set_ylim(bottom=0)
ax1.set_title("Analytical option values vs. Monte Carlo estimators (static simulation)")
wi = 1.0
ax2.bar(k_list - wi / 2, (anal_res - stat_res) / anal_res * 100, wi)
ax2.set_xlabel('strike')
ax2.set_ylabel('difference in %')
ax2.set_xlim(left=75, right=125)
plt.show(block = False)

## Now compare for dynamic simulation
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
ax1.plot(k_list, anal_res, 'b', label='analytical')
ax1.plot(k_list, dyna_res, 'ro', label='dynamic')
ax1.set_ylabel('European call option value')
ax1.set_title("Analytical option values vs. Monte Carlo estimators (dynamic simulation)")
ax1.legend(loc=0)
ax1.set_ylim(bottom=0)
wi = 1.0
ax2.bar(k_list - wi / 2, (anal_res - dyna_res) / anal_res * 100, wi)
ax2.set_xlabel('strike')
ax2.set_ylabel('difference in %')
ax2.set_xlim(left=75, right=125)
plt.show(block=False)

### Now let's look at American options - 
# consider the Least-squares regression for American Option Valuation
# min 1/I * sum (i=1:I) [Y_t_i - sum(d=1:D) (alpha_d_t * b_d(S_t_i) ) ]**2
# the function below implements this for both calls & puts

def gbm_mcs_amer(K, option='call'):
    ''' Valuation of American option in Black-Scholes-Merton
    by Monte Carlo simulation by LSM algorithm
    Parameters
    ==========
    K: float
    (positive) strike price of the option
    option: string
    type of the option to be valued ('call', 'put')
    Returns
    =======
    C0: float
    estimated present value of American call option
    '''
    dt = T / M
    df = math.exp(-r * dt)
    # simulation of index levels
    S = np.zeros((M + 1, I))
    S[0] = S0
    sn = gen_sn(M, I)
    for t in range(1, M + 1):
        S[t] = S[t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt
            + sigma * math.sqrt(dt) * sn[t])
    # case based calculation of payoff
    if option == 'call':
        h = np.maximum(S - K, 0)
    else:
        h = np.maximum(K - S, 0)
    # LSM algorithm
    V = np.copy(h)
    for t in range(M - 1, 0, -1):
        reg = np.polyfit(S[t], V[t + 1] * df, 7)
        C = np.polyval(reg, S[t])
        V[t] = np.where(C > h[t], V[t + 1] * df, h[t])
    # MCS estimator
    C0 = df * np.mean(V[1])
    return C0

# run the sim for call and for put option
print("The Dynamic Monte Carlo estimator value for the American call option: {}\n".format(gbm_mcs_amer(K=110., option='call')))

print("The Dynamic Monte Carlo estimator value for the American put option: {}\n".format(gbm_mcs_amer(K=110., option='put')))

# let's compare European and American options:
# The European option is considered a lower bound for American options
# the difference between the 2 are referred to as early exercise premium
# let's compare:

euro_res = []
amer_res = []
k_list = np.arange(80., 120.1, 5.)
for K in k_list:
    euro_res.append(gbm_mcs_dyna(K, 'put'))
    amer_res.append(gbm_mcs_amer(K, 'put'))
euro_res = np.array(euro_res)
amer_res = np.array(amer_res)

# plot to compare
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
ax1.plot(k_list, euro_res, 'b', label='European put')
ax1.plot(k_list, amer_res, 'ro', label='American put')
ax1.set_ylabel('call option value')
ax1.legend(loc=0)
ax1.set_title("European vs. American Monte Carlo estimators")
wi = 1.0
ax2.bar(k_list - wi / 2, (amer_res - euro_res) / euro_res * 100, wi)
ax2.set_xlabel('strike')
ax2.set_ylabel('early exercise premium in %')
ax2.set_xlim(left=75, right=125)
plt.show()
## This script is intended to show examples of Risk Management
## Taken from chapter 12, pages 383 - 392 of Reference book
import math
import numpy as np
import numpy.random as npr
from pylab import plt, mpl
import scipy.stats as scs

plt.style.use('seaborn-v0_8')
mpl.rcParams['font.family'] = 'serif'

# first let's look at Value at Risk:
# a model for assessing risk of a position/portfolio

# setup model for Black-Scholes-Merton formula
S0 = 100
r = 0.05
sigma = 0.25
T = 30 / 365.
I = 10000

# run evaluation using random values to model Brownian motion
ST = S0 * np.exp((r - 0.5 * sigma ** 2) * T +
    sigma * np.sqrt(T) * npr.standard_normal(I))

# calculate return & sort
R_gbm = np.sort(ST - S0)

# plot the data
#plt.figure(figsize=(10, 6))
plt.hist(R_gbm, bins=50)
plt.xlabel('absolute return')
plt.ylabel('frequency')
plt.title("Absolute profits and losses from simulation (geometric Brownian motion)")
plt.show(block=False)

percs = [0.01, 0.1, 1., 2.5, 5.0, 10.0]
var = scs.scoreatpercentile(R_gbm, percs)
print("30-Day Value at Risk Evaluation:")
print('%16s %16s' % ('Confidence Level', 'Value-at-Risk'))
print(33 * '-')
for pair in zip(percs, var):
    print('%16.2f %16.3f' % (100 - pair[0], -pair[1]))
print("\n")

# Let's look at another example - The jump diffusion model, define in valuations_examples.py script

delta = 0.25 # jump volatility
lamb = 0.75 # jump intensity
mu = -0.6 # mean jump size
M = 50

# run model
dt = 30. / 365 / M
rj = lamb * (math.exp(mu + 0.5 * delta ** 2) - 1)
S = np.zeros((M + 1, I))
S[0] = S0
sn1 = npr.standard_normal((M + 1, I))
sn2 = npr.standard_normal((M + 1, I))
poi = npr.poisson(lamb * dt, (M + 1, I))
for t in range(1, M + 1, 1):
    S[t] = S[t - 1] * (np.exp((r - rj - 0.5 * sigma ** 2) * dt
        + sigma * math.sqrt(dt) * sn1[t])
        + (np.exp(mu + delta * sn2[t]) - 1)
        * poi[t])
S[t] = np.maximum(S[t], 0)
R_jd = np.sort(S[-1] - S0)

# plot data
plt.figure(figsize=(10, 6))
plt.hist(R_jd, bins=50)
plt.xlabel('absolute return')
plt.ylabel('frequency')
plt.title('Absolute profits and losses from simulation (jump diffusion)')
plt.show(block=False)

# Now show VAR
percs = [0.01, 0.1, 1., 2.5, 5.0, 10.0]
var = scs.scoreatpercentile(R_jd, percs)
print("30-Day Value at Risk Evaluation of jump diffusion model:")
print('%16s %16s' % ('Confidence Level', 'Value-at-Risk'))
print(33 * '-')
for pair in zip(percs, var):
    print('%16.2f %16.3f' % (100 - pair[0], -pair[1]))


# compare both models directly
    percs = list(np.arange(0.0, 10.1, 0.1))
gbm_var = scs.scoreatpercentile(R_gbm, percs)
jd_var = scs.scoreatpercentile(R_jd, percs)

plt.figure(figsize=(10, 6))
plt.plot(percs, gbm_var, 'b', lw=1.5, label='GBM')
plt.plot(percs, jd_var, 'r', lw=1.5, label='JD')
plt.legend(loc=4)
plt.xlabel('100 - confidence level [%]')
plt.ylabel('value-at-risk')
plt.title("Value-at-risk for geometric Brownian motion and jump diffusion")
plt.ylim(ymax=0.0)
plt.show(block=False)


# now let's look at Credit Valuation Adjustments:
# Other important risk measures are the credit value-at-risk (CVaR) and the credit valuationadjustment (CVA)

# setup model again to show these:
S0 = 100.
r = 0.05
sigma = 0.2
T = 1.
I = 100000
ST = S0 * np.exp((r - 0.5 * sigma ** 2) * T
    + sigma * np.sqrt(T) * npr.standard_normal(I))

# define loss level
L = 0.5

# define probability of default
p = 0.01

# simulate data using poisson distribution
D = npr.poisson(p * T, I)

# limit default to first event
D = np.where(D > 1, 1, D)

# Now compare different VAR/CVAR methods:
print("Discounted average simulated value of the asset at time T = {}: {}\n".format(T,math.exp(-r * T) * np.mean(ST)))

CVaR = math.exp(-r * T) * np.mean(L * D * ST)
print("CVaR as the discounted average of the future losses in the case of a default at time T = {}: {}\n".format(T,CVaR))

S0_CVA = math.exp(-r * T) * np.mean((1 - L * D) * ST)
print("Discounted average simulated value of the asset at time T = {}, adjusted for the simulated losses from default: {}\n".format(T,S0_CVA))

S0_adj = S0 - CVaR
print("Current price of the asset adjusted by the simulated CVaR at time T = {}: {}\n".format(T,S0_adj))

# calculate the amount of loss events modeled in sim:
print("Number of default events and therewith loss events: {}\n".format(T,np.count_nonzero(L * D * ST)))

# plot the data
plt.figure(figsize=(10, 6))
plt.hist(L * D * ST, bins=50)
plt.xlabel('loss')
plt.ylabel('frequency')
plt.ylim(ymax=175)
plt.title("Losses due to risk-neutrally expected default (stock)")
plt.show(block=False)

# consider model using European call option:
K = 100.
hT = np.maximum(ST - K, 0)

C0 = math.exp(-r * T) * np.mean(hT)
print("The Monte Carlo estimator value for the European call option: {}\n".format(C0))

CVaR = math.exp(-r * T) * np.mean(L * D * hT)
print("The CVaR as the discounted average of the future losses in the case of a default: {}\n".format(CVaR))

C0_CVA = math.exp(-r * T) * np.mean((1 - L * D) * hT)
print("The Monte Carlo estimator value for the European call option, adjusted for the simulated losses from defaults: {}\n".format(C0_CVA))

# compare losses
print("The number of losses due to default: {}\n".format(np.count_nonzero(L * D * hT)))

print("The number of defaults: {}\n".format(np.count_nonzero(D)))

print("The number of cases for which the option expires worthless: {}\n".format(I - np.count_nonzero(hT)))

# and plot:
plt.figure(figsize=(10, 6))
plt.hist(L * D * hT, bins=50)
plt.xlabel('loss')
plt.ylabel('frequency')
plt.ylim(ymax=350)
plt.title("Losses due to risk-neutrally expected default (call option)")
plt.show()

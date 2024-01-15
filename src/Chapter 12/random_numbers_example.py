## This script is intended to show examples of random numbers
## Taken from chapter 12, pages 346 - 351 of Reference book
import numpy as np
import numpy.random as npr
from pylab import plt, mpl

plt.style.use('seaborn-v0_8')
mpl.rcParams['font.family'] = 'serif'

#Fixes the seed value for reproducibility and fixes the number of digits for printouts.
npr.seed(100)
np.set_printoptions(precision=4)

# set number of samples to pull from random generator
sample_size = 500

# sample from uniform distribution
rn1 = npr.rand(sample_size, 3)

# sample from integers - with lower bound 0 and upper bound 10
rn2 = npr.randint(0, 10, sample_size)

# sample from uniform distribution as well
rn3 = npr.sample(size=sample_size)

# Randomly sampled values from these 4 values in a list
a = [0, 25, 50, 75, 100]
rn4 = npr.choice(a, size=sample_size)

# make subplots to show random samples
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2,figsize=(10, 8))
ax1.hist(rn1, bins=25, stacked=True)
ax1.set_title('rand')
ax1.set_ylabel('frequency')
ax2.hist(rn2, bins=25)
ax2.set_title('randint')
ax3.hist(rn3, bins=25)
ax3.set_title('sample')
ax3.set_ylabel('frequency')
ax4.hist(rn4, bins=25)
ax4.set_title('choice')
plt.show(block = False)

## sample from different kinds of distributions:

# sample from standard normal distribution
rn1 = npr.standard_normal(sample_size)

# sample from normal distribution, centered at 100, with scale = 20
rn2 = npr.normal(100, 20, sample_size)

# sample from chi-squared distribution
rn3 = npr.chisquare(df=0.5, size=sample_size)

# sample from Poisson distribution
rn4 = npr.poisson(lam=1.0, size=sample_size)

# create subplots to show different samples
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2,figsize=(10, 8))
ax1.hist(rn1, bins=25)
ax1.set_title('standard normal')
ax1.set_ylabel('frequency')
ax2.hist(rn2, bins=25)
ax2.set_title('normal(100, 20)')
ax3.hist(rn3, bins=25)
ax3.set_title('chi square')
ax3.set_ylabel('frequency')
ax4.hist(rn4, bins=25)
ax4.set_title('Poisson')
plt.show()
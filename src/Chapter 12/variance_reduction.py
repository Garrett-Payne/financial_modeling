## This script is intended to show examples of variance reduction for stochastic modeling
## Taken from chapter 12, pages 372 - 375 of Reference book
import numpy as np
import numpy.random as npr



# example to show how statistics of a random sampling will be close to 0 mean & 1 std, but not exact.
# values generally will converce though
print("Values for numpy standard normal distributions:")
print('%15s %15s %15s' % ('Sample Size', 'Mean', 'Std. Deviation'))
print(31 * '-')
for i in range(1, 31, 2):
    npr.seed(100)
    nums = i ** 2 * 1000
    sn = npr.standard_normal(nums)
    print('%i %15.12f %15.12f' % (nums,sn.mean(), sn.std()))

print("\n")
## first method to reduce variance - antithetic variates
# this can be done by creating nomral distribution & then combining with the negative of each value
sn = npr.standard_normal(int(10000 / 2))
sn = np.concatenate((sn, -sn))

print("Values for antithetic variates:")
print('%15s %15s %15s' % ('Sample Size', 'Mean', 'Std. Deviation'))
print(31 * '-')
for i in range(1, 31, 2):
    npr.seed(1000)
    sn = npr.standard_normal(i ** 2 * int(10000 / 2))
    sn = np.concatenate((sn, -sn))
    print('%i %15.12f %15.12f' % (len(sn),sn.mean(), sn.std()))
print("\n")

## test out another method - moment matching
# this gives a sampling that corrects for first and second moments
sn = npr.standard_normal(10000)
sn_new = (sn - sn.mean()) / sn.std()

# creat function to do so
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


print("Values for moment method:")
print('%15s %15s %15s' % ('Sample Size', 'Mean', 'Std. Deviation'))
print(31 * '-')
for i in range(1, 31, 2):
    npr.seed(1000)
    M = i ** 2 * 1000
    sn =  gen_sn(M, 1, anti_paths=False, mo_match=True)
    print('%i %15.12f %15.12f' % (len(sn),sn.mean(), sn.std()))
print("\n")
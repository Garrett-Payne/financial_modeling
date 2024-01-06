## This script is intended to show procesing times of recursive pandas algorithm
## Taken from chapter 10, pages 305 - 307 of Reference book
import time
import numpy as np
import numba
import pyximport; pyximport.install(pyimport=True,language_level=3)
import pandas as pd
from pylab import plt 

# example is shown on how to implement the exponentially weighted moving average (EWMA)

## pythonic way

# define symbol to look for
sym = 'SPY'
# loat data
data_path = '../../data/tr_eikon_eod_data.csv'
data = pd.DataFrame(pd.read_csv(data_path, index_col=0, parse_dates=True)[sym]).dropna()

alpha = 0.25

# create new column that is a copy of the SPY data
data['EWMA'] = data[sym]

# calculate the new field in pythonic way (for loop)
t1 = time.time()
for t in zip(data.index, data.index[1:]):
    data.loc[t[1], 'EWMA'] = (alpha * data.loc[t[1], sym] +(1 - alpha) * data.loc[t[0], 'EWMA'])
t2 = time.time()
print("Time to calculate EWMA, pythonic way in dataframe: {} secs\n".format(t2-t1))


# plot the data
data[data.index > '2017-1-1'].plot(figsize=(10, 6))
plt.show(block=False)

# create pythonic function to calculate EWMA
def ewma_py(x, alpha): 
    y = np.zeros_like(x)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * x[i] + (1-alpha) * y[i-1] 
    return y

# test times to use function

t1 = time.time()
data['EWMA_PY'] = ewma_py(data[sym], alpha)
t2 = time.time()
print("Time to calculate EWMA, pythonic function (on dataframe): {} secs\n".format(t2-t1))

t1 = time.time()
data['EWMA_PY'] = ewma_py(data[sym].values, alpha)
t2 = time.time()
print("Time to calculate EWMA, pythonic function (on dataframe.values): {} secs\n".format(t2-t1))


## Now use numba to speed up function
ewma_nb = numba.jit(ewma_py)

# and test times
''' NOTE: the below will not wirk with python 3.12 - numba errored out since it could not directly
work on a pandas series. it needed the .values added to make it into an array
t1 = time.time()
data['EWMA_NB'] = ewma_nb(data[sym], alpha)
t2 = time.time()
print("Time to calculate EWMA, Numba function (on dataframe): {} secs\n".format(t2-t1))
'''

t1 = time.time()
data['EWMA_NB'] = ewma_nb(data[sym].values, alpha)
t2 = time.time()
print("Time to calculate EWMA, Numba function (on dataframe.values): {} secs\n".format(t2-t1))

## Test using cython function
import recursive_EWMA_cython

''' This also won't work in python 3.12 - cython function expects an array ,not a Pandas series
t1 = time.time()
data['EWMA_PY'] = recursive_EWMA_cython.ewma_cy(data[sym], alpha)
t2 = time.time()
print("Time to calculate EWMA, cython function (on dataframe): {} secs\n".format(t2-t1))
'''

t1 = time.time()
data['EWMA_PY'] = recursive_EWMA_cython.ewma_cy(data[sym].values, alpha)
t2 = time.time()
print("Time to calculate EWMA, cython function (on dataframe.values): {} secs\n".format(t2-t1))
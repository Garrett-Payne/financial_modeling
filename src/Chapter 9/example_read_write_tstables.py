## This script is intended to show procesing times for reading/writing to files using pytables
## Taken from chapter 9, pages 267 - 272 of Reference book
import time
import numpy as np
from pylab import plt
import tables as tb 
import datetime as dt
import pandas as pd
import tstables as tstab
import random

path = '../data/'
filename = path + 'pytab.h5'

# number of data points
no = 5000000 

# number of time series
co=3

# set interval for 1 year divided by minutes
interval=1./(12*30*24*60) 

# volatility coefficent
vol = 0.2

# create an Euler discretization model, and time how long it takes to create the data
t1 = time.time()
rn = np.random.standard_normal((no, co))
rn[0] = 0.0
paths = 100 * np.exp(np.cumsum(-0.5 * vol ** 2 * interval + vol * np.sqrt(interval) * rn, axis=0)) 
paths[0] = 100
t2 = time.time()
print("Time to create Euler discretization: {} secs\n".format(t2-t1))

# create date range array
dr = pd.date_range('2019-1-1', periods=no, freq='1s')

df = pd.DataFrame(paths, index=dr, columns=['ts1', 'ts2', 'ts3'])

# now plot the time series:


df[::100000].plot(figsize=(10, 6),title='Euler Discretization Models')
plt.show(block=False)


## Work with TsTables

# create new class for describing each column of data
class ts_desc(tb.IsDescription): 
    timestamp = tb.Int64Col(pos=0)
    ts1 = tb.Float64Col(pos=1) 
    ts2 = tb.Float64Col(pos=2) 
    ts3 = tb.Float64Col(pos=3)

# create and open new file
h5 = tb.open_file(path + 'tstab.h5', 'w') 

# create new ts object based upon class description above
ts = h5.create_ts('/', 'ts', ts_desc)

# time how long it takes to append data to ts object
t1 = time.time()
ts.append(df)
t2 = time.time()
print("Time to append data to new ts class dataset: {} secs\n".format(t2-t1) )

# setup limits for reading data
read_start_dt = dt.datetime(2019, 2, 1, 0, 0) 
read_end_dt = dt.datetime(2019, 2, 5, 23, 59)

# time how long it takes to read range of data from ts object
t1 = time.time()
rows = ts.read_range(read_start_dt, read_end_dt)
t2 = time.time()
print("Time to read subset of data: {} secs\n".format(t2-t1) )

h5.close()

# plot data
(rows[::500] / rows.iloc[0]).plot(figsize=(10, 6))
plt.show()

## Read in large dataset
h5 = tb.open_file(path + 'tstab.h5', 'r')
ts = h5.root.ts._f_get_timeseries()

t1 = time.time()
for _ in range(100):
    d = random.randint(1, 24)
    read_start_dt = dt.datetime(2019, 2, d, 0, 0, 0) 
    read_end_dt = dt.datetime(2019, 2, d + 3, 23, 59, 59) 
    rows = ts.read_range(read_start_dt, read_end_dt)
t2 = time.time()
print("Time to read large set of data: {} secs\n".format(t2-t1) )

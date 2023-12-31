## This script is intended to show procesing times for reading/writing to files using numpy
## Taken from chapter 9, pages 242 - 244 of Reference book

import numpy as np
import time

path = '../data/'

# create numpy array of datetimes
dtimes = np.arange('2019-01-01 10:00:00', '2025-12-31 22:00:00', dtype='datetime64[m]')

# define fields and datatypes for each field for a numpy array
dty = np.dtype([('Date', 'datetime64[m]'), ('No1', 'f'), ('No2', 'f')])

# instantiate numpy array with the field specified above & number of rows equal to the len of dtimes
data = np.zeros(len(dtimes), dtype=dty)

# assign the 'Date' field to be the dtimes array
data['Date'] = dtimes

# create a numpy array of length len(dtimes) x 2 with random values and assign into data
a = np.random.standard_normal((len(dtimes), 2)).round(4)
data['No1'] = a[:, 0]
data['No2'] = a[:, 1]

print("Number of bytes of data struct: {}\n".format(data.nbytes))

# save off array to a file, and time how long it takes
t1 = time.time()
np.save(path + 'array', data)
t2 = time.time()
print("Time to save full numpy array: {} secs".format(t2-t1))

# read in file that was written & time how long it takes
t1 = time.time()
d2 = np.load(path + 'array.npy')
t2 = time.time()
print("Time to read in full numpy array: {} secs\n".format(t2-t1))

# create very large array, time how long it takes to create
t1 = time.time()
data = np.random.standard_normal((10000, 6000)).round(4)
t2 = time.time()
print("Time to create 10000 x 6000 random numpy array: {} secs".format(t2-t1))

# save off new array again
t1 = time.time()
np.save(path + 'array', data)
t2 = time.time()
print("Time to save off 10000 x 6000 random numpy array: {} secs".format(t2-t1))

# read in new array again
t1 = time.time()
d2 = np.load(path + 'array.npy')
t2 = time.time()
print("Time to read in 10000 x 6000 random numpy array: {} secs\n".format(t2-t1))

# note that it's much more efficient to read in a saved off numpy array than create one over and over again.
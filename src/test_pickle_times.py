## This script is intended to show procesing times for using pickle library to read/write files.
## Taken from chapter 9, pages 233 - 235 of Reference book
import pickle 
import numpy as np
import time
from random import gauss

# Make a large array of random values
a = [gauss(1.5, 2) for i in range(1000000)] 
# define where to write data to
path = '../data/'

pkl_file = open(path + 'data.pkl', 'wb') # open file in writeable, binary format
t1 = time.time()
pickle.dump(a, pkl_file)
t2 = time.time()
print("Time to write to pickle file: {:4} seconds".format(t2-t1))
pkl_file.close()

# Now test loading in file
pkl_file = open(path + 'data.pkl', 'rb')
t1 = time.time()
b = pickle.load(pkl_file)
t2 = time.time()
print("Time to read pickle file: {:4} seconds".format(t2-t1))

# check to verify a and b are the same (or are similar enough)
print("a and b are close: " + str(np.allclose(np.array(a), np.array(b))))


pkl_file = open(path + 'data.pkl', 'wb')
t1 = time.time()
pickle.dump(np.array(a), pkl_file)
t2 = time.time()
print("Time to dump a as a numpy array: {:4}".format(t2-t1))

t1 = time.time()
pickle.dump(np.array(a) ** 2, pkl_file)
t2 = time.time()
print("Time to dump 2*a as a numpy array: {:4}".format(t2-t1))

pkl_file.close()

pkl_file = open(path + 'data.pkl', 'rb')
'''
t1 = time.time()
b = pickle.load(pkl_file)
t2 = time.time()
print("Time to read numpy array pickle file: {:4} seconds".format(t2-t1))

print("length of read in file: {}".format(len(b)))
print("length of numpy a array: {}".format(len(a)))
'''

x = pickle.load(pkl_file)
y = pickle.load(pkl_file)
pkl_file = open(path + 'data.pkl', 'wb') 
pickle.dump({'x': x, 'y': y}, pkl_file) 
pkl_file.close()
pkl_file = open(path + 'data.pkl', 'rb') 
data = pickle.load(pkl_file) 
pkl_file.close()
for key in data.keys():
    print(key, data[key][:4])

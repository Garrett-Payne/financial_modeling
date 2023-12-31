## This script is intended to show procesing times for reading/writing to sql database files and pandas files
## Taken from chapter 9, pages 245 - 252 of Reference book
import pandas as pd
import sqlite3 as sq3
import time
import numpy as np
from pylab import plt

path = '../data/'
filename = path + 'numbers'

# create large array of random data to be used to write to file
data = np.random.standard_normal((1000000, 5)).round(4)

# create new connection
con = sq3.Connection(filename + '.db')

# create string containing query to run
query = 'CREATE TABLE numbers (No1 real, No2 real, No3 real, No4 real, No5 real)'
q = con.execute
qm = con.executemany
q(query)

# write data into sql table named numbers
t1 = time.time()
qm('INSERT INTO numbers VALUES (?, ?, ?, ?, ?)', data) 
con.commit()
t2 = time.time()
print("Time to write to sql database: {} secs\n".format(t2-t1))

# read data from sql table named numbers
t1 = time.time()
temp = q('SELECT * FROM numbers').fetchall()
t2 = time.time()
print("Time to read in data from sql numbers database: {} secs\n".format(t2-t1))

# make new query string to select data from numbers table
query = 'SELECT * FROM numbers WHERE No1 > 0 AND No2 < 0' 
t1 = time.time()
res = np.array(q(query).fetchall()).round(3)
t2 = time.time()
print("Time to read in query with sql conneciton into numpy array: {} seconds\n".format(t2-t1))

# now plot the data that's been read in
res = res[::100] 
plt.figure(figsize=(10, 6))
plt.plot(res[:, 0], res[:, 1], 'ro')
plt.title("First 100 values in SQL table")
plt.show(block=False)

# Test reading in data using pandas directly
t1 = time.time()
data = pd.read_sql('SELECT * FROM numbers', con)
t2 = time.time()
print("Time to read in query with sql conneciton into pandas array: {} seconds\n".format(t2-t1))

# test time to filter data using pandas conditional statements
t1 = time.time()
d2 = data[(data['No1'] > 0) & (data['No2'] < 0)]
t2 = time.time()
print("Time to read filter query in loaded data dataframe: {} seconds\n".format(t2-t1))

# time to test another query using direct connection with q 
t1 = time.time()
q = '(No1 < -0.5 | No1 > 0.5) & (No2 < -1 | No2 > 1)' 
res = data[['No1', 'No2']].query(q)
t2 = time.time()
print("Time to read filter query in from sql database: {} seconds\n".format(t2-t1))

# create another plot to show
plt.figure(figsize=(10, 6)) 
plt.plot(res['No1'], res['No2'], 'ro')
plt.title('filtered data scatterplot')
plt.show(block=False)

## Work with h5 files:
# this will create an HDF5 for writing to
h5s = pd.HDFStore(filename + '.h5s', 'w')
t1 = time.time()
h5s['data'] = data
t2 = time.time()
print("Time to write data to an HDF5 file: {} seconds\n".format(t2-t1))

# make sure to close out file
h5s.close()

# read in file again
t1 = time.time()
h5s = pd.HDFStore(filename + '.h5s', 'r')
data_ = h5s['data']
t2 = time.time()
print("Time to read data in from an HDF5 file: {} seconds\n".format(t2-t1))

# make sure to close connection
h5s.close()

# test writing to/from csv files
t1 = time.time()
data.to_csv(filename + '.csv')
t2 = time.time()
print("Time to write pandas data to a csv file: {} seconds\n".format(t2-t1))

# reading in
t1 = time.time()
df = pd.read_csv(filename + '.csv')
t2 = time.time()
print("Time to read pandas data from a csv file: {} seconds\n".format(t2-t1))

plt.figure()
df[['No1', 'No2', 'No3', 'No4']].hist(bins=20, figsize=(10, 6),title='Pandas dataframe data')
plt.show(block=False)

# test reading/writing with Excel files
t1 = time.time()
data[:100000].to_excel(filename + '.xlsx')
t2 = time.time()
print("Time to write first 100000 rows of pandas data to an excel file: {} seconds\n".format(t2-t1))

t1 = time.time()
df = pd.read_excel(filename + '.xlsx', 'Sheet1')
t2 = time.time()
print("Time to read data from excel file: {} seconds\n".format(t2-t1))

# plot cumulative data from df dataframe
plt.figure()
df.cumsum().plot(figsize=(10, 6))
plt.show()
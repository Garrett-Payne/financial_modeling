## This script is intended to show procesing times for reading/writing to sql database files
## Taken from chapter 9, pages 239 - 241 of Reference book
import sqlite3 as sq3
import datetime
import time
import numpy as np

# define where to write data to
path = '../data/'
con = sq3.connect(path + 'numbs.db')

# query to use to create a new sqlite table
query = 'CREATE TABLE numbs (Date date, No1 real, No2 real)'

# below line executres the query
con.execute(query)

# commites changes to database
con.commit()

# below code makes a variable based on the connection
q = con.execute

# this will et the info about the database
q('SELECT * FROM sqlite_master').fetchall() 

# collect current time
now = datetime.datetime.now()

# write 3 values into the numbs database
q('INSERT INTO numbs VALUES(?, ?, ?)', (now, 0.12, 7.3))


np.random.seed(100)
data = np.random.standard_normal((10000, 2)).round(4)
t1 = time.time()
for row in data:
    now = datetime.datetime.now()
    q('INSERT INTO numbs VALUES(?, ?, ?)', (now, row[0], row[1])) 
    con.commit()

t2 = time.time()
print("time to write values to database: {} seconds\n".format(t2-t1))

# This selects the first 4 values from the database table
aa = q('SELECT * FROM numbs').fetchmany(4)
print("Fetching 4 values from numbs database:")
print(aa)

# This selects the first 4 values from the database table where the no1 column is greater than 0.5
aa = q('SELECT * FROM numbs WHERE no1 > 0.5').fetchmany(4)
print("Fetching 4 values from numbs database where no1 > 0.5:")
print(aa)


pointer = q('SELECT * FROM numbs')
print(" Fetching one value at a time using for loop")
for i in range(3): 
    print(pointer.fetchone())

# collects all remaining rows
rows = pointer.fetchall()
print("Next 3 rows have been fetched:")
print(rows[:3])

# Delete table
q('DROP TABLE IF EXISTS numbs')

aa = q('SELECT * FROM sqlite_master').fetchall()
print("Remaining values in database:")
print(aa)

# close out sq3 connection
con.close()


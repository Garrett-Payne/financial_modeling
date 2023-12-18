''' This is an example based upon Chapter 8 of reference book: pages 206-222'''
import numpy as np
import pandas as pd
from pylab import mpl, plt
import time



# define filename of csv to read in
filename = '../data/tr_eikon_eod_data.csv'
data = pd.read_csv(filename, index_col=0, parse_dates=True)

# FIGURE 1
# view the data in multple subplots
data.plot(subplots=True, title="Overview of all data in Eikon CSV file")
plt.show(block=False)
'''
NOTE: to have figures non-blocking (i.e. all pop up as code runs, rather than having 
to exit out of a plot for the next one), you must put 'block=False in the plt.show() command
'''

# description for each column in the dataframe
instruments = ['Apple Stock', 'Microsoft Stock',
                           'Intel Stock', 'Amazon Stock', 'Goldman Sachs Stock',
                           'SPDR S&P 500 ETF Trust', 'S&P 500 Index',
                           'VIX Volatility Index', 'EUR/USD Exchange Rate',
                           'Gold Price', 'VanEck Vectors Gold Miners ETF',
                           'SPDR Gold Trust']

# Print each column and associated description

# Note: I want to test time for iterations in a for loop with using zip and integer iterator
# test zip method
t1_zip = time.time()
for ric, name in zip(data.columns, instruments):
    print('{:8s} | {}'.format(ric, name))
t2_zip = time.time()
print("Time to run for loop using zip: {} secs\n".format(t2_zip-t1_zip))

# test iterator method
'''
t1_range = time.time()
for i in range(len(data.columns)):
    print('{:8s} | {}'.format(data.columns[i], instruments[i]))
t2_range = time.time()
print("Time to run for loop using zip: {} secs\n".format(t2_range-t1_range))
'''
# Seems like iterator using range() is faster than zip() - can uncomment to test for yourself.

# FIGURE 2
# Plot diffs (difference between two index values (i minus i-1))
data.diff().plot(subplots=True, title="Diff of of all data in Eikon CSV file")
plt.show(block=False)

# FIGURE 3
# Plot percentage change between 2 index values
data.pct_change().plot(subplots=True, title="Percentage change of all data in Eikon CSV file")
plt.show(block=False)

# FIGURE 4
plt.figure()
data.pct_change().mean().plot(kind='bar', figsize=(10, 6),title="Percent Change Mean Values")
plt.show(block=False)

# calculate log return values for each data source in data
rets = np.log(data / data.shift(1))

# FIGURE 5
# create plot that shows cumulative sum
rets.cumsum().plot(figsize=(10, 6),title='Log Returns Cumulative Sum')
plt.show(block=False)

# FIGURE 6
# create plot that shows cumulative sum - exponential scale ( the .appy(np.exp) method calculates each as exponential)
rets.cumsum().apply(np.exp).plot(figsize=(10, 6),title='Log Returns Cumulative Sum')
plt.show(block=False)

# FIGURE 7
# example of resampling - downsampling to 1 month - note that graph is less noiser not that is down-sampled to 1 month
rets.cumsum().apply(np.exp).resample('1m', label='right').last( ).plot(figsize=(10, 6),title='Log Returns Cumulative Sum (Downsampled to 1 Month)')
plt.show(block=False)


## Calculating Rolling Statistics ##
sym = 'AAPL.O'
data = pd.DataFrame(data[sym]).dropna() # re-allocates data dataframe to be only AAPL data, also removes all NAN values
window = 20 # amount of values to use in rolling window
data['min'] = data[sym].rolling(window=window).min()
data['mean'] = data[sym].rolling(window=window).mean()
data['std'] = data[sym].rolling(window=window).std()
data['median'] = data[sym].rolling(window=window).median()
data['max'] = data[sym].rolling(window=window).max()
data['ewma'] = data[sym].ewm(halflife=0.5, min_periods=window).mean()

# plot rolling mean, min, max values over time
ax = data[['min', 'mean', 'max']].iloc[-200:].plot( figsize=(10, 6), style=['g--', 'r--', 'g--'], lw=0.8,title='AAPL Stock Rolling Statistics')
data[sym].iloc[-200:].plot(ax=ax, lw=2.0)
plt.show(block=False)

# Calculate SMAs (Simple Moving Averages)
data['SMA1'] = data[sym].rolling(window=42).mean()
data['SMA2'] = data[sym].rolling(window=252).mean()

ax = data[[sym,'SMA1','SMA2']].plot(figsize=(10,6),title='AAPL Stock Simple Moving Averages')
ax.legend(['Daily','42 Day','252 Day'])
plt.show(block=False)

data.dropna(inplace=True)
data['positions'] = np.where(data['SMA1'] > data['SMA2'], 1, -1)
ax = data[[sym, 'SMA1', 'SMA2', 'positions']].plot(figsize=(10, 6), secondary_y='positions')
ax.get_legend().set_bbox_to_anchor((0.25, 0.85))
plt.show()

'''Note: Last plt.show() command in a script with multiple plots shown must NOT have 'block=False' in it'''
''' This is an example based upon Chapter 8 of reference book: pages 222-228'''
import numpy as np
import pandas as pd
from pylab import mpl, plt
import time

# read in data
raw = pd.read_csv('../data/tr_eikon_eod_data.csv',index_col=0, parse_dates=True)

# filter down to SPX & VIX tickers
data = raw[['.SPX', '.VIX']].dropna()

# plot basic dataset
data.plot(subplots=True, figsize=(10, 6),title="SPX an VIX Timelines")
plt.show(block=False)

# filter all data until befor end of year 2012, and plot VIX as right y
data.loc[:'2012-12-31'].plot(secondary_y='.VIX', figsize=(10, 6))
plt.show(block=False)

# Calculate log returns
rets = np.log(data / data.shift(1))
rets.dropna(inplace=True)

# Plot the log returns over time
rets.plot(subplots=True, figsize=(10, 6),title='Log Returns')
plt.show(block=False)

# plot the scatter plot & Kernel Density Estimate (histogram)
pd.plotting.scatter_matrix(rets, alpha=0.2,diagonal='hist', hist_kwds={'bins': 35}, figsize=(10, 6))
plt.show(block=False)

# ordinary least-squares (OLS) regression analysis
# calculate linear regression using numpy polyfit function
reg = np.polyfit(rets['.SPX'], rets['.VIX'], deg=1)

# plot the log returns, along with the approximated curve fit
ax = rets.plot(kind='scatter', x='.SPX', y='.VIX', figsize=(10, 6),title='Compare Returns with Least Squares Model')
ax.plot(rets['.SPX'], np.polyval(reg, rets['.SPX']), 'r', lw=2)
ax.legend(['Log Returns','Curve Fit'])
plt.show(block=False)

# Analyze correlation
plt.figure()
ax = rets['.SPX'].rolling(window=252).corr( rets['.VIX']).plot(figsize=(10, 6),title='Correlation')
ax.axhline(rets.corr().iloc[0, 1], c='r')
ax.legend(['Rolling Correlation','Static Correlation'])
plt.show()
'''Note: Last plt.show() command in a script with multiple plots shown must NOT have 'block=False' in it'''
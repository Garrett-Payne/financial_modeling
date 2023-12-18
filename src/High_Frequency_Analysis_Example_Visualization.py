''' This is an example based upon Chapter 8 of reference book: pages 228-230'''
import numpy as np
import pandas as pd
from pylab import mpl, plt
import time

# Read in tick data
tick = pd.read_csv('../data/fxcm_eur_usd_tick_data.csv',index_col=0, parse_dates=True)

# calculating means for each row and plot - mean is the middle between ask and bid prices
tick['Mid'] = tick.mean(axis=1) 
ax = tick['Mid'].plot(figsize=(10, 6),title='Middle Value Over Time')

# resample data to 5 min intervals
tick_resam = tick.resample(rule='5min', label='right').last()
tick_resam['Mid'].plot(figsize=(10, 6))
ax.legend(['sub-second values','5 Minute Sampling'])
plt.show()
'''Note: Last plt.show() command in a script with multiple plots shown must NOT have 'block=False' in it'''
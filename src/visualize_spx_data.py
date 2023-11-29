import numpy as np 
import pandas as pd
from pylab import plt, mpl

plt.style.use('seaborn-v0_8') 
mpl.rcParams['font.family'] = 'serif' 
data = pd.read_csv('../data/tr_eikon_eod_data.csv', index_col=0, parse_dates=True)
data = pd.DataFrame(data['.SPX']) 
data.dropna(inplace=True)
data['rets'] = np.log(data / data.shift(1))
data['vola'] = data['rets'].rolling(252).std() * np.sqrt(252)
data[['.SPX', 'vola']].plot(subplots=True, figsize=(10, 6))
plt.show()
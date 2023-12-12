''' Example based on Chapter 7 of reference book - pages 195 - 203'''

import cufflinks as cf
import pandas as pd
import plotly.offline as plyo

# Load in data file
raw = pd.read_csv('../data/fxcm_eur_usd_eod_data.csv', index_col=0, parse_dates=True)
quotes = raw[['OpenAsk', 'HighAsk', 'LowAsk', 'CloseAsk']] # filter down to these 4 columns
quotes = quotes.iloc[-60:] # Take last 60 values

# Sets up a QuantFig object
qf = cf.QuantFig( quotes,
                  title='EUR/USD Exchange Rate',
                  legend='top',
                  name='EUR/USD')

# Creates the plot
plyo.iplot( qf.iplot(asFigure=True),
            image='png',
            filename='qf_01' )
#NOTE: default plyo will open browser to add a plot

## add bollinger bands to the qf object
qf.add_bollinger_bands(periods=15, boll_std=2)
# create new plot (in new tab) to show candle plots with bollinger bands
plyo.iplot(qf.iplot(asFigure=True), image='png',
filename='qf_02' )

# add RSI to plot - adds as a subplot below main plot
qf.add_rsi(periods=14, showbands=False)
# create another plot
plyo.iplot(qf.iplot(asFigure=True), image='png', filename='qf_03' )
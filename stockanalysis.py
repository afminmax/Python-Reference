## IMPORT PANDAS 
import pandas as pd
import datetime
import pandas_datareader.data as web # imports web data reader api
from pandas import Series, DataFrame

## DEFINE THE TIMEFRAME (FOR OUR ANALYSIS, THIS DECADE)
start = datetime.datetime(2010,1,1)
end = datetime.datetime(2019,9,5)

## SELECT THE STOCK AND THE LOCATION OF THE DATA
df = web.DataReader("GE", 'yahoo', start, end)
df.tail() # display the last 5 records

## MOVING AVERAGE ('ROLLING MEAN')
close_price = df['Adj Close']
moving_avg = close_price.rolling(window=100).mean() # compute rolling price for last 100 days


## IMPORT MATPLOTLIB TO CHART
%matplotlib inline
import matplotlib.pyplot as plt
from matplotlib import style

# Adjusting the size of matplotlib
import matplotlib as mpl
mpl.rc('figure', figsize=(8,7))
mpl.__version__

# Adjusting the style of matplotlib
style.use('ggplot')

close_price.plot(label='GE')
moving_avg.plot(label='mavg')
plt.legend()

## DETERMINING RETURNS
returns = close_price / close_price.shift(1) - 1
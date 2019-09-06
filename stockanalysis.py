## IMPORT PANDAS 
import pandas as pd
import datetime
import math
import numpy as np
import scipy
import sklearn
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
returns = close_price / close_price.shift(1) - 1 ## need to look at this more
returns.plot(label = 'return')

## COMPARING RETURNS OF SIMILAR COMPANIES
dfcomps = web.DataReader(['MMM', 'EMR', 'ROK', 'UTX','GE'],'yahoo',start=start,end=end)['Adj Close']
dfcomps.tail() # show data

## CORRELATION ANALYSIS
retscomp = dfcomps.pct_change()
corr = retscomp.corr()
corr # show data

plt.scatter(retscomp.MMM, retscomp.GE) # correlating GE and 3M
plt.xlabel('Returns MMM')
plt.ylabel('Returns GE')

# Applying Kernel Density Estimate
#pd.scatter_matrix(retscomp, diagonal='kde', figsize=(10, 10)); #deprecated syntax

pd.plotting.scatter_matrix(retscomp, diagonal='kde', figsize=(10, 10));

## PLOTTING HEAT MAPS
# The lighter the color, the more correlated the stocks are
plt.imshow(corr, cmap='hot', interpolation='none')
plt.colorbar()
plt.xticks(range(len(corr)), corr.columns)
plt.yticks(range(len(corr)), corr.columns)
# Problem with heat maps is correlations but no causality. You might just be seeing trends in sectors or the market.

## CALCULATING RATE OF RETURN AND RISK
plt.scatter(retscomp.mean(), retscomp.std())
plt.xlabel('Expected returns')
plt.ylabel('Risk')
for label, x, y in zip(retscomp.columns, retscomp.mean(), retscomp.std()):
    plt.annotate(
        label, 
        xy = (x, y), xytext = (20, -20),
        textcoords = 'offset points', ha = 'right', va = 'bottom',
        bbox = dict(boxstyle = 'round,pad=0.5', fc = 'gray', alpha = 0.5),
        arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))

## PREDICTING THE STOCK PRICE - STEP 1: PRE-PROCESSING
## A number of pre-processing steps are required before feeding data into the models

# 1. First we need high-low %, and % change
dfreg = df.loc[:,['Adj Close','Volume']]
dfreg['HL_PCT'] = (df['High'] - df['Low']) / df['Close'] * 100.0
dfreg['PCT_change'] = (df['Close'] - df['Open']) / df['Open'] * 100.0
dfreg.tail()

# 2. Drop missing value
dfreg.fillna(value=-99999, inplace=True)

# 3. We want to separate 1 percent of the data to forecast
forecast_out = int(math.ceil(0.01 * len(dfreg))) # requires math! (import math)

# 4. Separating the label here, we want to predict the AdjClose
forecast_col = 'Adj Close'
dfreg['label'] = dfreg[forecast_col].shift(-forecast_out)
X = np.array(dfreg.drop(['label'], 1)) # requires numpy! (import numpy as np)
X

# 5. Scale the X so that everyone can have the same distribution for linear regression
from sklearn.preprocessing import scale
X = scale(X)
X

# 6. Finally We want to find the Data Series of late X and early X (train) for model generation and evaluation
X_lately = X[-forecast_out:]
X = X[:-forecast_out]
X

# 7. Separate label and identify it as y
y = np.array(dfreg['label'])
y = y[:-forecast_out]

# 8. Separation of training and testing of model by cross validation train test split
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

## PREDICTING THE STOCK PRICE - SIMPLE LINEAR ANALYSIS AND QUADRATIC DISCRIMINANT ANALYSIS
# Load up the scikit models
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Linear regression
clfreg = LinearRegression(n_jobs=-1)
clfreg.fit(X_train, y_train)

# Quadratic Regression 2
clfpoly2 = make_pipeline(PolynomialFeatures(2), Ridge())
clfpoly2.fit(X_train, y_train)

# Quadratic Regression 3
clfpoly3 = make_pipeline(PolynomialFeatures(3), Ridge())
clfpoly3.fit(X_train, y_train)

## NEED TO PUT SOME PLOTTING HERE?

## KNN {K NEAREST NEIGHBOR}
# KNN Regression
clfknn = KNeighborsRegressor(n_neighbors=2)
clfknn.fit(X_train, y_train)

confidencereg = clfreg.score(X_test, y_test)
('The linear regression confidence is ', confidencereg)

confidencepoly2 = clfpoly2.score(X_test,y_test)
('The quadratic regression 2 confidence is ', confidencepoly2)

confidencepoly3 = clfpoly3.score(X_test,y_test)
('The quadratic regression 3 confidence is ', confidencepoly3)

confidenceknn = clfknn.score(X_test, y_test)
('The knn regression confidence is ', confidenceknn)

# Plotting the prediction
last_date = dfreg.iloc[-1].name
last_unix = last_date
next_unix = last_unix + datetime.timedelta(days=1)

for i in forecast_set:
    next_date = next_unix
    next_unix += datetime.timedelta(days=1)
    dfreg.loc[next_date] = [np.nan for _ in range(len(dfreg.columns)-1)]+[i]
dfreg['Adj Close'].tail(500).plot()
dfreg['Forecast'].tail(500).plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()





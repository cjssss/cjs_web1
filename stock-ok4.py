# 그래프가 안 맞음 확인 필요
from pandas_datareader import data
import datetime
import fix_yahoo_finance as yf

yf.pdr_override()
start_date = '2008-01-01'
name = '036570.KS'
nc = data.get_data_yahoo(name, start_date)
print(nc.head(3))
print(nc.tail(3))

import pandas as pd
import matplotlib.pyplot as plt
## %matplotlib inline
plt.show()

nc['Close'].plot(figsize=(12, 6), grid=True) # graph

nc_trunc = nc[:'2016-12-31']
#print(nc_trunc.head(3))
df = pd.DataFrame({'ds':nc_trunc.index, 'y':nc_trunc['Close']})
df.reset_index(inplace=True)
del df['Date']
df.head(3)

from fbprophet import Prophet
m = Prophet()
m.fit(df);

future = m.make_future_dataframe(periods=365*2)
print(future.tail(3))

forecast = m.predict(future)
forecast[['ds', 'yhat']].tail(3)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(3)

#m.plot(forecast)
#m.plot_components(forecast)

plt.figure(figsize=(12, 6))
plt.plot(nc.index, nc['Close'], label='real')
plt.plot(forecast['ds'], forecast['yhat'], label='forecast')
plt.grid()
plt.legend()
plt.show()


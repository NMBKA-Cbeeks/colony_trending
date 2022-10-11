#!/usr/bin/env python3.8

import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from statsmodels.tsa.seasonal import seasonal_decompose

colony_file = "../data/Carniolan Layens_combined_readings_2022-10-10T20_33_00.281Z.csv" 

df = pd.read_csv( colony_file )

df.Unix_Time = pd.to_datetime(df.Unix_Time, unit='s', utc = True )

#df = df.set_index('Unix_Time')
#df.head()

#print( df )

df.plot( x = 'Unix_Time', y = 'Scaled_Weight' )
plt.show()


scaled_signal = [ ( item, time )  for ( item, time ) in zip( df.Scaled_Weight, df.Unix_Time ) if str(item) != 'nan' ]

scaled_weight = [ i[ 0 ] for i in scaled_signal ]
scaled_time = [ i[ 1 ] for i in scaled_signal ]

weight_df = pd.DataFrame( { 'data' : scaled_weight }, index = scaled_time )
weight_df.index=pd.to_datetime(weight_df.index)

detrended = signal.detrend( scaled_weight )

detrended_df = pd.DataFrame( detrended, index = scaled_time )

detrended_df.plot()
plt.legend( ["detrended"] )
plt.show()

res = seasonal_decompose( weight_df, model = 'additive',
                          extrapolate_trend = 'freq',
                          period = 365 )

seasonal_detrended = scaled_weight - res.trend

seasonal_detrended_df = pd.DataFrame( seasonal_detrended, index = scaled_time )
seasonal_detrended_df.plot()
plt.legend( ["seasonal detrended"] )
plt.show()

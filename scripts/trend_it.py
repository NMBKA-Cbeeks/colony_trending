#!/usr/bin/env python3.8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from scipy import signal
import matplotlib.dates as mdates
from statsmodels.tsa.seasonal import seasonal_decompose

lbs_per_bee = 0.00025 * 5

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

#print( weight_df )
burn_date = pd.to_datetime( '2022-10-03 00:00:00.000000000', utc = True )
#print( type( burn_date ) )

burn_weight_df = weight_df[ weight_df.index < burn_date  ]
print( burn_weight_df )
x = mdates.date2num( burn_weight_df.index )

burn_coeff = np.polyfit(x, burn_weight_df.data, 1)
print( burn_coeff )

print( "Burn Rate: " + "{rate:.2f}".format( 
        rate = burn_coeff[ 0 ] ) + "lbs / day" )

detrended = signal.detrend( scaled_weight )

detrended_df = pd.DataFrame( detrended, index = scaled_time )

detrended_df.plot()
plt.legend( ["detrended"] )
plt.show()

res = seasonal_decompose( weight_df, model = 'multiplicative',
                          extrapolate_trend = 'freq',
                          period = 365 )

seasonal_detrended = ( scaled_weight - res.trend ) / lbs_per_bee

seasonal_detrended_df = pd.DataFrame( seasonal_detrended, index = scaled_time )
seasonal_detrended_df.plot()
plt.legend( ["Active Foragers"] )
plt.show()

#!/usr/bin/env python3.8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from scipy import signal
import matplotlib.dates as mdates
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.interpolate import CubicSpline as cspline

lbs_per_bee = 0.00025

colony_file = "../data/Carniolan Layens_combined_readings_2023-04-12T23_23_31.434Z.csv" 

df = pd.read_csv( colony_file )

df.Unix_Time = pd.to_datetime(df.Unix_Time, unit='s', utc = True )

#df = df.set_index('Unix_Time')
#df.head()

#print( df )

df.plot( x = 'Unix_Time', y = 'Scaled_Weight' )
plt.xlabel( 'Date (YYYY-MM-DD)' )
plt.ylabel( 'Weight (lbs.)' )
plt.legend( [ 'Layens' ] )
plt.savefig( "../figs/layens-weight-overall.png" )
plt.show()


scaled_signal = [ ( item, time )  for ( item, time ) in zip( df.Scaled_Weight, df.Unix_Time ) if str(item) != 'nan' ]

scaled_weight = [ i[ 0 ] for i in scaled_signal ]
scaled_time = [ i[ 1 ] for i in scaled_signal ]

weight_df = pd.DataFrame( { 'data' : scaled_weight }, index = scaled_time )
weight_df.index=pd.to_datetime(weight_df.index)
#print( weight_df )

# Valid data period

# Create fit of overall trend
x = mdates.date2num( weight_df.index )
trend_coeff = np.polyfit(x, weight_df.data, 2)
x_range = np.arange( min( x ), max( x ), 
                ( max( x ) - min( x ) ) / 100 )
poly = np.poly1d( trend_coeff )
df.plot( x = 'Unix_Time', y = 'Scaled_Weight', color = 'darkorange')
plt.plot( mdates.num2date( x_range ), poly( x_range ), label = "Trend" )
plt.xlabel( 'Date (YYYY-MM-DD)' )
plt.ylabel( 'Weight (lbs.)' )
plt.legend( ["Layens","Model"] )
plt.savefig( "../figs/layens-winter-model.png" )
plt.show()

fig, ax = plt.subplots()
plt.plot( mdates.num2date( x_range ), poly( x_range ), label = "Model" )
plt.xlabel( 'Date (YYYY-MM-DD)' )
plt.ylabel( 'Weight (lbs.)' )
ax.set_xticklabels( mdates.num2date( x_range ), rotation = 30 ) 
plt.legend( loc = 'best' )
plt.savefig( "../figs/layens-winter-model-only.png" )
plt.show()

# Burn rate line fitting for Late Sept data
burn_date_end = pd.to_datetime( '2023-04-08 00:00:00.000000000', utc = True )
burn_date_start = pd.to_datetime( '2022-08-28 00:00:00.000000000', utc = True )
burn_weight_df = weight_df[ weight_df.index > burn_date_start  ]
x_burn = mdates.date2num( burn_weight_df.index )
burn_coeff = np.polyfit(x_burn, burn_weight_df.data, 1)
poly_burn = np.poly1d( burn_coeff )
burn_weight_df.plot()
plt.plot( mdates.num2date( x_burn ), poly_burn( x_burn ), label = 'Fit' )
plt.xlabel( 'Date (YYYY-MM-DD)' )
plt.ylabel( 'Weight (lbs.)' )
plt.legend( ["Layens","Confinement Model"] )
plt.savefig( "../figs/layens-winter-burn.png" )
plt.show()
        
print( "Burn Rate: " + "{rate:.2f}".format( 
        rate = burn_coeff[ 0 ] ) + "lbs / day" )

detrended = signal.detrend( scaled_weight )

detrended_df = pd.DataFrame( detrended, index = scaled_time )

detrended_df.plot()
plt.legend( ["detrended"] )
plt.show()

# Active foragers: these calculations still need more work,
# initial plot  for the presentation.  I need to think a bit more
# about the dips in weight.  This really shoudl prolly be the inverse
# over the X-axis of what is there now ... hmmm -kbf
res = seasonal_decompose( weight_df, model = 'multiplicative',
                          extrapolate_trend = 'freq',
                          period = 365 )

seasonal_detrended = ( scaled_weight - res.trend ) / ( lbs_per_bee * 4 )
seasonal_detrended_df = pd.DataFrame( seasonal_detrended, index = scaled_time )
seasonal_detrended_df.plot()
plt.xlabel( 'Date (YYYY-MM-DD)' )
plt.ylabel( 'Active Foragers (bees)' )
plt.legend( ["Layens"] )
plt.subplots_adjust( left = 0.15 )
plt.axhline( y = 0, color = 'orange' )
plt.savefig( "../figs/layens-winter-foragers.png" )
plt.show()

fig, ax = plt.subplots()
x_range = mdates.date2num( seasonal_detrended_df.index )
x2_range = mdates.date2num( df.Unix_Time )
l1 = ax.plot( mdates.num2date( x_range ), seasonal_detrended_df.values,
         label = "Foragers" )
ax_temp = ax.twinx()
l2 = ax_temp.plot( mdates.num2date( x2_range ), df["Temperature"], 
                  color = 'darkorange', label = 'Temp.' ) 
ax_temp.set_ylim( [ 0, 300 ] )
plt.xlabel( 'Date (YYYY-MM)' )
ax.set_ylabel( 'Active Foragers (bees)' )
ax_temp.set_ylabel( 'Ambiant Temp (F)' )
#ax.set_xticklabels( mdates.num2date( x_range ), rotation = 30 ) 

# added these three lines
fig.subplots_adjust( left = 0.15 )
lns = l1+l2
labs = [l.get_label() for l in lns]

ax.legend(lns, labs, loc='best')
plt.savefig( "../figs/layens-winter-model-temp.png" )
plt.show()

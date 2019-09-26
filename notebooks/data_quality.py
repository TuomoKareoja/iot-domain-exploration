#%%

import copy
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pmdarima
import seaborn as sns
import sklearn
import statsmodels
from dotenv import find_dotenv, load_dotenv
from IPython.core.interactiveshell import InteractiveShell

# Setting styles
InteractiveShell.ast_node_interactivity = 'all'
sns.set(style='whitegrid', color_codes=True)
%matplotlib inline

#%%

# loading the data

data_raw_path = os.path.join('data', 'raw', 'submeters.csv')
data_raw = pd.read_csv(data_raw_path, parse_dates=[['Date', 'Time']])

# There are no missing values
data_raw.dtypes
data_raw.isnull().sum()
data_raw.describe()
data_raw.head()
data_raw.shape


#%%

# Id has doubles (same id in multiple datasets)
data_raw.duplicated(subset=['id']).sum()
# Date_Time does not have duplicates, so no overlap between the datasets
data_raw.duplicated(subset=['Date_Time']).sum()

#%%

# checking that the Date_Time covers the whole range minute by minute
# and does not have holes

index_start = data_raw.Date_Time.min()
index_end = data_raw.Date_Time.max()
index_new = pd.date_range(index_start, index_end, freq='min')

# There are holes so we have to broaden the index
print("observations in full index: ",index_new.shape[0])
print("observations in current index: ",data_raw.index.shape[0])

# Setting Date_Time as index
data_raw.set_index('Date_Time', inplace=True)

# broadening the index
data_raw = data_raw.asfreq('min')
print("observations in corrected index: ",data_raw.index.shape[0])

#%%

# Checking the amount of missing values
data_raw.isnull().sum()

#%%

# The missing values seem to be only from certain dates
data_raw[['id']].isnull().groupby([data_raw.index.year]).sum().plot()
data_raw[['id']].isnull().groupby([data_raw.index.month]).sum().plot()
data_raw[['id']].isnull().groupby([data_raw.index.date]).sum().plot()
data_raw[['id']].isnull().groupby([data_raw.index.time]).sum().plot()

#%%

# Finding the biggest holes in the dataseries

# new dataframe with only id
data_missing = data_raw[['id']].copy()

# check that first number is not missing
data_missing.iloc[0,:]

# add new colum that is 1 when id not null
data_missing['not_missing'] = np.where(data_missing.id.isnull(), 0, 1)

# shift values of new column on forward
data_missing['not_missing_shifted'] = data_missing.not_missing.shift(1)

# keep only values where id is missing
data_missing = data_missing[data_missing.id.isnull()]

# drop other columns that shifted marker
data_missing.drop(columns=['id', 'not_missing', 'not_missing'], inplace=True)

# calculate rolling sum to create id fro
data_missing['missing_period_id'] = data_missing.not_missing_shifted.cumsum()

# drop unnecessary column
data_missing.drop(columns=["not_missing_shifted"], inplace=True)


# Name the episode by with the start and end time
data_missing['datetime'] = data_missing.index
data_missing = data_missing.groupby(['missing_period_id']).agg(['min', 'max', 'count'])
data_missing.columns = data_missing.columns.get_level_values(0)
data_missing.columns = ['start', 'end', 'length_min']
data_missing['start'] = data_missing.start.apply(lambda x: x.strftime('%Y-%m-%d %H:%M'))
data_missing['end'] = data_missing.end.apply(lambda x: x.strftime('%Y-%m-%d %H:%M'))
data_missing['period'] = data_missing.start + ' to ' + data_missing.end
data_missing.reset_index(level=0, inplace=True)
data_missing.drop(columns=['start', 'end', 'missing_period_id'],inplace=True)
data_missing.set_index(data_missing.period, inplace=True)
data_missing.drop(columns=['period'],inplace=True)

# plot missing periods
data_missing.plot.bar(figsize=(12,9))

# There are two three periods that last for over 10 hours.
# This is a problem for interpolating missing values because
# the data has day level seasonality.

# If there is need to add in missing values. We have to keep these
# problems in mind

#%%

# Are there periods in Global active power that seem unrealistic

data_raw_daily = data_raw.resample('D').mean()
ax = sns.lineplot(x=data_raw_daily.index, y=data_raw_daily.Global_active_power)
ax.set_ylim(bottom=0)
plt.show()

# the beginning of 2017 seems too high.

#%%

# What are the ends of the index?
data_raw.index.min()
data_raw.index.max()

# And we can see that the index extends
# to 2006 december and 2011 october. We would want to keep only full years
# to make the analysis easier, as then we don't have to keep partial
# years in mind when plotting yearly trends

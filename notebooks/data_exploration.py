#%%

import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import os
import copy
from dotenv import find_dotenv, load_dotenv
from IPython.core.interactiveshell import InteractiveShell
from src.data.load_data import load_processed_data

# Setting styles
InteractiveShell.ast_node_interactivity = 'all'
sns.set(style='whitegrid', color_codes=True)
%matplotlib inline

#%%

# load data
data = load_processed_data()

#%% [markdown]

## Questions:

# 1. What is the average hourly consumption
# ..* What is the average hourly consumption
# 1. Does the average con

 
#%%

data_test = data.loc['2007']

#%%

# draw day trend
# global_columns = ['Global_active_power', 'Global_reactive_power', 'Global_intensity']
weekday_order = ["Monday", 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
figure_dims = (12,10)

fig, (ax1, ax2, ax3) = plt.subplots(3,1,figsize=figure_dims)

sns.pointplot(x='weekday', y='Global_active_power', data=data_test, order=weekday_order, ax=ax1)
sns.pointplot(x='weekday', y='Global_reactive_power', data=data_test, order=weekday_order, ax=ax2)
sns.pointplot(x='weekday', y='Global_intensity', data=data_test, order=weekday_order, ax=ax3)
ax1.set(xlabel='Weekday', ylabel=None, title="Global Active Power Weekday Trend")
ax2.set(xlabel='Weekday', ylabel=None, title="Global Reactive Power Weekday Trend")
ax3.set(xlabel='Weekday', ylabel=None, title="Global Intensity Weekday Trend")
# What are the proper limits considering the phenomena we are measuring
ax1.set_ylim(bottom=0)
ax2.set_ylim(bottom=0)
ax3.set_ylim(bottom=0)
plt.show()

# TODO we need to show min and max also

#%%

# TODO monty plots

# TODO yearly plots
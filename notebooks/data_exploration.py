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

# 1. What are trends in all measures yearly, monthly, daily and hourly
# 2. How is the electricity use spread between different measures yearly, monthly, daily and hourly

#%%

# defining functions to plot trends

def perc05(x):
    return np.percentile(x, q=5)

def perc95(x):
    return np.percentile(x, q=95)
 
def draw_trend(x, y, data, xlabel, ylabel, title, order=None, rotation=0):
    fig, ax = plt.subplots()
    sns.pointplot(x=x, y=y, data=data, ci=None, estimator=np.mean, color='green', order=order, ax=ax)
    sns.pointplot(x=x, y=y, data=data, ci=None, estimator=perc95, color='red', order=order, ax=ax)
    sns.pointplot(x=x, y=y, data=data, ci=None, estimator=perc05, color='blue', order=order, ax=ax)
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.set_ylim(bottom=0)
    plt.xticks(rotation=rotation)
    plt.show()

def draw_distribution(measures, labels, x, data, xlabel, titlepart, order=None, rotation=0):
    fig, ax = plt.subplots()
    colors = ["blue", "green", "red", 'black']
    sns.barplot(x=x, y="Global_active_power", data=data, ci=None, color='pink', order=order, ax=ax)
    ax2 = ax.twinx()
    for i, (measure, label) in enumerate(zip(measures, labels)):
        sns.pointplot(x=data[x], y=data[measure].div(data['Global_active_power']).multiply(100), label=label, ci=None, color=colors[i], order=order, ax=ax2)
    ax.set(ylabel="Total Energy Use")
    ax.set_ylim(bottom=0)
    ax2.set(ylabel="% of Total Energy Use", title="Percentage of " + titlepart + " Electricity Use")
    ax2.set_ylim(bottom=0)
    plt.xticks(rotation=rotation)
    plt.legend(labels=labels)
    plt.show()

#%%

weekday_order = ["Monday", 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

month_order = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]

#%%

# Plotting trends

measures = ["Global_active_power", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3", 'unmeasured']
xlabels = ["Global Active Power", "Submeter 1", "Submeter 2", "Submeter 3", 'Unmeasured']

title_suffix = " Yearly Trends"
for measure, xlabel in zip(measures, xlabels):
    draw_trend("year", measure, data=data, xlabel=xlabel, ylabel=None, title=xlabel + title_suffix)

title_suffix = " Monthly Trends"
for measure, xlabel in zip(measures, xlabels):
    draw_trend("month", measure, data=data, xlabel=xlabel, ylabel=None, title=xlabel + title_suffix, order=month_order, rotation=90)

title_suffix = " Daily Trends"
for measure, xlabel in zip(measures, xlabels):
    draw_trend("weekday", measure, data=data, xlabel=xlabel, ylabel=None, title=xlabel + title_suffix, order=weekday_order, rotation=90)

title_suffix = " Hourly Trends"
for measure, xlabel in zip(measures, xlabels):
    draw_trend("hour", measure, data=data, xlabel=xlabel, ylabel=None, title=xlabel + title_suffix)

#%%

# Plotting distribution between meters

measures = ["unmeasured", "Sub_metering_3", "Sub_metering_2", 'Sub_metering_1']
xlabels = ["Unmeasured", "Submeter 3", "Submeter 2", 'Submeter 1']

draw_distribution(measures=measures, labels=xlabels, x='year', data=data, xlabel='Year', titlepart='Yearly')
draw_distribution(measures=measures, labels=xlabels, x='month', data=data, xlabel='Month', titlepart='Monthly', order=month_order, rotation=90)
draw_distribution(measures=measures, labels=xlabels, x='weekday', data=data, xlabel='Weekday', titlepart='Daily', order=weekday_order, rotation=90)
draw_distribution(measures=measures, labels=xlabels, x='hour', data=data, xlabel='hour', titlepart='Hourly')

#%%

# heatmaps of the correlations between different measures

correlations_pearson = data[["Global_active_power", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3", 'unmeasured']].corr(method='pearson')
correlations_spearman = data[["Global_active_power", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3", 'unmeasured']].corr(method='spearman')

sns.heatmap(correlations_pearson, annot=True, cmap='Reds')
plt.show()
sns.heatmap(correlations_spearman, annot=True, cmap='Reds')
plt.show()

#%%

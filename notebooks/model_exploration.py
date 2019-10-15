#%%

import copy
import os

import holidays
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pmdarima as pm
import seaborn as sns
import sklearn
from dotenv import find_dotenv, load_dotenv
from fbprophet import Prophet
from IPython.core.interactiveshell import InteractiveShell
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.data.load_data import load_processed_data

# Setting styles
InteractiveShell.ast_node_interactivity = "all"
sns.set(style="whitegrid", color_codes=True)

#%%

data = load_processed_data()

#%% hour level predictions for the month

# TODO
# Pipeline for Sarimax crossvalidation
# Pipeline for Prophet crossvalidation
# Pipeline for Holt-Winters crossvalidation
# Errors calculated hourly daily and for the full month
# Visualizations for error and for hour level predictions
# add temperature and dew point
# Remember to limit predictions to zero!

#%% day level predictions for the month

# TODO
# Resample the data
# Pipeline for Sarimax crossvalidation
# Pipeline for Prophet crossvalidation
# Pipeline for Holt-Winters crossvalidation
# Errors calculated hourly daily and for the full month
# Visualizations for error and for day level predictions
# add temperature and dew point
# Remember to limit predictions to zero!

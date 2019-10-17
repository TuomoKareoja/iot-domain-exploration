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

# limiting the data
data = load_processed_data()
data = data["2007":"2010"].resample("1D").mean()

#%%

# pm.plot_acf(data["Global_active_power"], lags=365 * 2, zero=False)
# pm.plot_pacf(data["Global_active_power"], lags=365 * 2, zero=False)
# print("The p-value for the ADF test is ", adfuller(data["Global_active_power"])[1])

# pm.plot_acf(data["Global_active_power"].diff(1).dropna(), lags=365 * 2, zero=False)
# pm.plot_pacf(data["Global_active_power"].diff(1).dropna(), lags=365 * 2, zero=False)
# print(
#     "The p-value for the ADF test is ",
#     adfuller(data["Global_active_power"].diff(1).dropna())[1],
# )

# pm.plot_acf(data["Global_active_power"].diff(365).dropna(), lags=365 * 2, zero=False)
# pm.plot_pacf(data["Global_active_power"].diff(365).dropna(), lags=365 * 2, zero=False)
# print(
#     "The p-value for the ADF test is ",
#     adfuller(data["Global_active_power"].diff(365).dropna())[1],
# )


# pm.plot_acf(
#     data["Global_active_power"].diff(1).diff(7).dropna(), lags=365 * 2, zero=False
# )
# pm.plot_pacf(
#     data["Global_active_power"].diff(1).diff(7).dropna(), lags=365 * 2, zero=False
# )
# print(
#     "The p-value for the ADF test is ",
#     adfuller(data["Global_active_power"].diff(1).diff(7).dropna())[1],
# )

# pm.plot_acf(
#     data["Global_active_power"].diff(1).diff(7).diff(365).dropna(),
#     lags=365 * 2,
#     zero=False,
# )
# pm.plot_pacf(
#     data["Global_active_power"].diff(1).diff(7).diff(365).dropna(),
#     lags=365 * 2,
#     zero=False,
# )
# print(
#     "The p-value for the ADF test is ",
#     adfuller(data["Global_active_power"].diff(1).diff(7).diff(365).dropna())[1],
# )

# pm.plot_acf(
#     data["Global_active_power"].diff(1).diff(2).diff(7).diff(365).dropna(),
#     lags=365 * 2,
#     zero=False,
# )
# pm.plot_pacf(
#     data["Global_active_power"].diff(1).diff(2).diff(7).diff(365).dropna(),
#     lags=365 * 2,
#     zero=False,
# )
# print(
#     "The p-value for the ADF test is ",
#     adfuller(data["Global_active_power"].diff(1).diff(2).diff(7).diff(365).dropna())[1],
# )

# pm.plot_acf(
#     data["Global_active_power"].diff(7).diff(365).dropna(), lags=365 * 2, zero=False
# )
# pm.plot_pacf(
#     data["Global_active_power"].diff(7).diff(365).dropna(), lags=365 * 2, zero=False
# )
# print(
#     "The p-value for the ADF test is ",
#     adfuller(data["Global_active_power"].diff(7).diff(365).dropna())[1],
# )

# Quite difficult to parse so we leave this to automation and time

#%%

# setting the seasonal pattern to 365 days
# and not giving the differencing options as the dataset is now much smaller

data = load_processed_data()
data = data["2007":"2010"].resample("1D").mean()

#%%
# estimate number of seasonal differences using a Canova-Hansen test
D = pm.arima.utils.nsdiffs(data["Global_active_power"], m=7, max_D=4, test="ch")
d = pm.arima.utils.ndiffs(data["Global_active_power"], test="adf")

# shows seasonal 2 and normal 0
# We know that weekday matters so differentiate the seventh day, but no others


#%%

data = load_processed_data()
data = data["2009":"2009"].resample("1D").mean()


#%%

autoarima_model = pm.auto_arima(
    data["Global_active_power"],
    trace=1,
    seasonal=True,
    m=7,
    start_p=2,
    start_q=1,
    max_p=7,
    max_q=7,
    start_P=0,
    start_Q=0,
    max_P=4,
    max_Q=4,
)

#%%

print(autoarima_model.summary())

# Fit ARIMA: order=(2, 1, 1) seasonal_order=(1, 0, 1, 7); AIC=2060.210, BIC=2087.490, Fit time=1.990 seconds
# Total fit time: 28.350 seconds
#                                  Statespace Model Results
# =========================================================================================
# Dep. Variable:                                 y   No. Observations:                  365
# Model:             SARIMAX(1, 1, 1)x(1, 0, 1, 7)   Log Likelihood               -1023.136
# Date:                           Thu, 17 Oct 2019   AIC                           2058.272
# Time:                                   20:17:28   BIC                           2081.655
# Sample:                                        0   HQIC                          2067.565
#                                            - 365
# Covariance Type:                             opg
# ==============================================================================
#                  coef    std err          z      P>|z|      [0.025      0.975]
# ------------------------------------------------------------------------------
# intercept      0.0004      0.002      0.229      0.819      -0.003       0.004
# ar.L1          0.1057      0.048      2.199      0.028       0.011       0.200
# ma.L1         -0.8936      0.022    -40.773      0.000      -0.937      -0.851
# ar.S.L7        0.9761      0.018     53.835      0.000       0.941       1.012
# ma.S.L7       -0.8710      0.050    -17.439      0.000      -0.969      -0.773
# sigma2        15.9289      1.001     15.916      0.000      13.967      17.890
# ===================================================================================
# Ljung-Box (Q):                       25.59   Jarque-Bera (JB):                27.16
# Prob(Q):                              0.96   Prob(JB):                         0.00
# Heteroskedasticity (H):               0.62   Skew:                             0.34
# Prob(H) (two-sided):                  0.01   Kurtosis:                         4.15
# ===================================================================================


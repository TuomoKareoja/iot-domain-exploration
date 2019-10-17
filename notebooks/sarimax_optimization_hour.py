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

#%%

# Determining the differencing manually to speed up model evaluation

pm.plot_acf(data["Global_active_power"], lags=24 * 1, zero=False)
pm.plot_pacf(data["Global_active_power"], lags=24 * 1, zero=False)
print("The p-value for the ADF test is ", adfuller(data["Global_active_power"])[1])

pm.plot_acf(data["Global_active_power"].diff(1).dropna(), lags=24 * 1, zero=False)
pm.plot_pacf(data["Global_active_power"].diff(1).dropna(), lags=24 * 1, zero=False)
print(
    "The p-value for the ADF test is ",
    adfuller(data["Global_active_power"].diff(1).dropna())[1],
)

pm.plot_acf(data["Global_active_power"].diff(24).dropna(), lags=24 * 1, zero=False)
pm.plot_pacf(data["Global_active_power"].diff(24).dropna(), lags=24 * 1, zero=False)
print(
    "The p-value for the ADF test is ",
    adfuller(data["Global_active_power"].diff(24).dropna())[1],
)

pm.plot_acf(
    data["Global_active_power"].diff(1).diff(24).dropna(), lags=24 * 1, zero=False
)
pm.plot_pacf(
    data["Global_active_power"].diff(1).diff(24).dropna(), lags=24 * 1, zero=False
)
print(
    "The p-value for the ADF test is ",
    adfuller(data["Global_active_power"].diff(1).diff(24).dropna())[1],
)

#%%

# liming the data
data = data["2008"]

#%%

ar_opt = [2, 3, 4, 5]
dif_opt = [1]
ma_opt = [0, 1]

seas_ar_opt = [1]
seas_dif_opt = [1]
seas_ma_opt = [0]
seas_period_opt = [24]

columns = [
    "ar",
    "dif",
    "ma",
    "seas_ar",
    "seas_dif",
    "seas_ma",
    "seas_period",
    "AIC",
    "BIC",
]

results_df = pd.DataFrame(columns=columns)
index_to_write = 0

for ar in ar_opt:
    for dif in dif_opt:
        for ma in ma_opt:
            for seas_ar in seas_ar_opt:
                for seas_dif in seas_dif_opt:
                    for seas_ma in seas_ma_opt:
                        for seas_period in seas_period_opt:

                            sarimax_model = SARIMAX(
                                data["Global_active_power"],
                                order=(ar, dif, ma),
                                seasonal_order=(
                                    seas_ar,
                                    seas_dif,
                                    seas_ma,
                                    seas_period,
                                ),
                            )
                            sarimax_res = sarimax_model.fit()
                            print(sarimax_res.summary())
                            results_df.loc[index_to_write] = [
                                ar,
                                dif,
                                ma,
                                seas_ar,
                                seas_dif,
                                seas_ma,
                                seas_period,
                                sarimax_res.aic,
                                sarimax_res.bic,
                            ]

                            index_to_write += 1

#%%

print(results_df)

#%%

# The results. Its seems that we are getting nowhere by just searching ourself

# ar  dif   ma  seas_ar  seas_dif  seas_ma  seas_period  AIC            BIC
# 2   1     0   1        1         0        24           68652.273667   68680.585016
# 2   1     1   1        1         0        24           67410.763779   67446.152964
# 3   1     0   1        1         0        24           68509.763657   68545.152842
# 3   1     1   1        1         0        24           67407.390401   67449.857424
# 4   1     0   1        1         0        24           68375.843619   68418.310641
# 4   1     1   1        1         0        24           67408.931022   67458.475881
# 5   1     0   1        1         0        24           68254.633784   68304.178644
# 5   1     1   1        1         0        24           67410.733237   67467.355933

#%%

# Pyramid arima to find reasonable parameters in automated way

autoarima_model = pm.auto_arima(
    data["Global_active_power"],
    d=1,
    D=0,
    trace=1,
    seasonal=True,
    m=24,
    start_p=2,
    start_q=1,
    max_p=2,
    max_q=1,
    start_P=2,
    start_Q=1,
)

# converged to:
# Fit ARIMA: order=(2, 1, 1) seasonal_order=(2, 0, 1, 24); AIC=64556.023, BIC=64612.667, Fit time=460.527 seconds

#%%

optimized_model = SARIMAX(
    data["Global_active_power"], order=(2, 1, 1), seasonal_order=(2, 0, 1, 24)
)
optimized_res = optimized_model.fit()

#%%

print(optimized_res.summary())

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
# 2006 has some weird ratings, so lets just use data from 2007 onwards
data = data["2007":"2008"]

#%%

target_variable = "Global_active_power"
# 24 * 30 = 30 days = 1 month
test_size = 24 * 30
tests = 2

#%%

# A dataframe that contains the correct measurements for the target
# variable for all test periods and one a length of one test_size begore
# them. That means that this dataframe can be used to check insample predictions
# and all test predictions

results = data[[target_variable]][-(test_size * (tests + 1)) :]
results["type"] = ["insample"] * test_size + [
    "test_{}".format(test_number)
    for _ in range(test_size)
    for test_number in range(1, tests + 1)
]

#%%


def prophet_cv(data, target_variable, test_size, tests, results):

    for test_number in range(1, tests + 1):

        test_split = len(data) - test_size * test_number

        df_train = data[:test_split][[target_variable]]

        # preparing data for prophet
        df_train.reset_index(level=0, inplace=True)
        df_train.columns = ["ds", "y"]

        model_prophet = Prophet()
        model_prophet.fit(df_train)
        prophet_future = model_prophet.make_future_dataframe(
            periods=test_size, freq="H"
        )
        prophet_prediction = model_prophet.predict(prophet_future)

        # limiting low predictions to zero
        prophet_prediction["yhat"] = np.where(
            prophet_prediction["yhat"] < 0, 0, prophet_prediction["yhat"]
        )

        # for the first test take also the insample predictions
        if test_number == 1:
            predictions = prophet_prediction["yhat"][-(2 * test_size) :].to_list()
        else:
            predictions = (
                predictions + prophet_prediction["yhat"][-(test_size):].to_list()
            )

    results["prophet"] = predictions

    return results


#%%

results = prophet_cv(
    data=data,
    target_variable=target_variable,
    test_size=test_size,
    tests=tests,
    results=results,
)

#%%


def sarimax_cv(data, target_variable, test_size, tests, results):

    for test_number in range(1, tests + 1):

        test_split = len(data) - test_size * test_number

        df_train = data[:test_split][[target_variable]]

        # see sarimax_optimization.py for finding the hyperparameters
        model_sarimax = SARIMAX(
            df_train["Global_active_power"],
            order=(2, 1, 1),
            seasonal_order=(2, 0, 1, 24),
        )
        res = model_sarimax.fit()

        # for the first test take also the insample predictions
        if test_number == 1:
            sarimax_prediction = res.get_prediction(
                start=len(df_train) - test_size, end=len(df_train) + test_size
            )
            # setting the floor to 0
            predictions = [
                value if value > 0 else 0 for value in sarimax_prediction.predicted_mean
            ]

        else:
            sarimax_prediction = res.get_prediction(
                start=len(df_train), end=len(df_train) + test_size
            )
            # setting the floor to 0
            predictions = predictions + [
                value if value > 0 else 0 for value in sarimax_prediction.predicted_mean
            ]

    results["sarimax"] = predictions

    return results


#%%

results = sarimax_cv(
    data=data,
    target_variable=target_variable,
    test_size=test_size,
    tests=tests,
    results=results,
)

#%%

results.plot()

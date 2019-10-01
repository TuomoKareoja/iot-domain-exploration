#%%

import copy
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from dotenv import find_dotenv, load_dotenv
from fbprophet import Prophet
from IPython.core.interactiveshell import InteractiveShell
from sklearn.metrics import mean_squared_error
import holidays

from src.data.load_data import load_processed_data

# Setting styles
InteractiveShell.ast_node_interactivity = "all"
sns.set(style="whitegrid", color_codes=True)

#%%

# load data
data = load_processed_data()
data.head()

#%%

data.isnull().sum()
data.dtypes

#%%


def compare_models(data, variable, test_size):

    test_split = len(data) - test_size

    # simple model using mean electricity use by month, weekday and hour
    data_train = data[:test_split][[variable]]
    data_train_grouped = (
        data_train[[variable]]
        .groupby(
            [data_train.index.month, data_train.index.weekday, data_train.index.hour]
        )
        .mean()
    )
    data_train_grouped.index.names = ["month", "weekday", "hour"]

    data_test = pd.DataFrame(
        data={
            "month": data[test_split:].index.month,
            "weekday": data[test_split:].index.weekday,
            "hour": data[test_split:].index.hour,
        },
        index=data[test_split:].index,
    )

    mean_grouped_predictions = data_test.join(
        data_train_grouped, how="left", on=["month", "weekday", "hour"]
    )[variable]

    # preparing data for prophet
    df = data[variable].reset_index(level=0)
    df.columns = ["ds", "y"]

    df_train = df[:test_split]
    df_test = df[test_split:]

    m_simple = Prophet()
    m_simple.fit(df_train)
    future_simple = m_simple.make_future_dataframe(periods=test_size, freq="H")
    forecast_simple = m_simple.predict(future_simple)
    # limiting low predictions to zero
    forecast_simple["yhat"] = np.where(
        forecast_simple["yhat"] < 0, 0, forecast_simple["yhat"]
    )
    forecast_simple["yhat_lower"] = np.where(
        forecast_simple["yhat_lower"] < 0, 0, forecast_simple["yhat_lower"]
    )
    forecast_simple["yhat_upper"] = np.where(
        forecast_simple["yhat_upper"] < 0, 0, forecast_simple["yhat_upper"]
    )
    forecast_plot_simple = m_simple.plot(forecast_simple)
    component_plot_simple = m_simple.plot_components(forecast_simple)

    # using inbuilt holidays because this automatically applies to predictions also
    m_holiday = Prophet()
    m_holiday.add_country_holidays(country_name="FRA")
    m_holiday.fit(df_train)
    future_holiday = m_holiday.make_future_dataframe(periods=test_size, freq="H")
    forecast_holiday = m_holiday.predict(future_holiday)
    # limiting low predictions to zero
    forecast_holiday["yhat"] = np.where(
        forecast_holiday["yhat"] < 0, 0, forecast_holiday["yhat"]
    )
    forecast_holiday["yhat_lower"] = np.where(
        forecast_holiday["yhat_lower"] < 0, 0, forecast_holiday["yhat_lower"]
    )
    forecast_holiday["yhat_upper"] = np.where(
        forecast_holiday["yhat_upper"] < 0, 0, forecast_holiday["yhat_upper"]
    )
    forecast_plot_holiday = m_holiday.plot(forecast_holiday)
    component_plot_holiday = m_holiday.plot_components(forecast_holiday)

    # calculate rmse

    df_test.y.describe()
    print(
        "Mean RMSE: ",
        mean_squared_error(df_test.y, np.repeat(df_train.y.mean(), len(df_test))),
    )
    print(
        "Mean grouped RMSE: ", mean_squared_error(df_test.y, mean_grouped_predictions)
    )
    print(
        "Simple Prophet: ",
        mean_squared_error(df_test.y, forecast_simple.yhat[test_split:]),
    )
    print(
        "Holiday Prophet: ",
        mean_squared_error(df_test.y, forecast_holiday.yhat[test_split:]),
    )


#%% [markdown]

# ## Global Active Power

#%%

compare_models(data=data, variable="Global_active_power", test_size=31 * 24)

#%% [markdown]

# ## Sub Meter 1

# Consumption is distinctly divided into on and off state and prophet and
# any of the other measures cannot predict these

#%%

compare_models(data=data, variable="Sub_metering_1", test_size=31 * 24)

#%% [markdown]

# ## Sub Meter 2

# Consumption is mostly small but has extreme spikes. Model does not capture these
# and fails to perform better than simple average


#%%

compare_models(data=data, variable="Sub_metering_2", test_size=31 * 24)
#%% [markdown]

# ## Sub Meter 3

# Consumption is distinctly divided into on and off state and prophet and
# any of the other measures cannot predict these. Highest true values never predicted

#%%

compare_models(data=data, variable="Sub_metering_3", test_size=31 * 24)

#%% [markdown]

# ## Unmeasured Consumption

# model gets the trend right, but does not perform better than mean grouped by month, weekday and hour

#%%

compare_models(data=data, variable="unmeasured", test_size=31 * 24)

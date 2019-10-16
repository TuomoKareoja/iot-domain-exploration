#%%

import copy
import os

import holidays
import matplotlib.pyplot as plt
import plotly.graph_objects as go
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
from statsmodels.tsa.api import ExponentialSmoothing
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.data.load_data import load_processed_data

#%% Setting styles

sns.set(style="whitegrid", color_codes=True)
col_red = "#D16BA5"
col_blue = "#86A8E7"
col_magenta = "#3ACAC0"
template = "plotly_white"

img_width = 700
img_height = 500
img_scale = 2

margin_l = 50
margin_r = 0
margin_b = 50
margin_t = 50
margin_pad = 0

#%% Setting save path

fig_path = os.path.join("reports", "figures")

#%%

data = load_processed_data()
# 2006 has some weird ratings, so lets just use data from 2007 onwards
data = data["2007"]

#%%

# GLOBALS

target_variable = "Global_active_power"
# 24 * 30 = 30 days = 1 month
test_size = 24 * 30
tests = 2
fig_path = os.path.join("reports", "figures")
img_graph_name = "model_comparison_timeseries"
img_table_name = "model_comparison_table"

#%%


def prophet_cv(results, freq):

    for test_number in range(1, tests + 1):

        test_split = len(data) - test_size * test_number

        df_train = data[:test_split][[target_variable]]

        # preparing data for prophet
        df_train.reset_index(level=0, inplace=True)
        df_train.columns = ["ds", "y"]

        model_prophet = Prophet()
        model_prophet.fit(df_train)
        prophet_future = model_prophet.make_future_dataframe(
            periods=test_size, freq=freq
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


def sarimax_cv(results, order, seasonal_order):

    for test_number in range(1, tests + 1):

        test_split = len(data) - test_size * test_number

        df_train = data[:test_split][[target_variable]]

        # see sarimax_optimization.py for finding the hyperparameters
        model_sarimax = SARIMAX(
            df_train[target_variable], order=order, seasonal_order=seasonal_order
        )
        res = model_sarimax.fit()

        # for the first test take also the insample predictions
        # we need to remove one from test_size as end is also included
        if test_number == 1:
            pred = res.get_prediction(
                start=len(df_train) - test_size,
                end=len(df_train) + test_size - 1,
                dynamic=test_size,
            )
            # setting the floor to 0
            predictions = [value if value > 0 else 0 for value in pred.predicted_mean]

        else:
            pred = res.get_prediction(
                start=len(df_train), end=len(df_train) + test_size - 1, dynamic=0
            )
            # setting the floor to 0
            predictions = predictions + [
                value if value > 0 else 0 for value in pred.predicted_mean
            ]

    results["sarimax"] = predictions

    return results


def holtwinters_cv(results, seasonal_periods):

    for test_number in range(1, tests + 1):

        test_split = len(data) - test_size * test_number

        df_train = data[:test_split][[target_variable]]

        res = ExponentialSmoothing(
            df_train,
            seasonal_periods=seasonal_periods,
            trend="add",
            seasonal="add",
            damped=True,
        ).fit(use_boxcox=True)

        # for the first test take also the insample predictions
        if test_number == 1:
            predictions = [
                value if value > 0 else 0 for value in res.fittedvalues[-test_size:]
            ] + [value if value > 0 else 0 for value in res.forecast(test_size)]

        else:
            predictions = predictions + [
                value if value > 0 else 0 for value in res.forecast(test_size)
            ]

    results["holtwinters"] = predictions

    return results


def aggregate_df(df, agg):
    df = df.resample(agg).mean()
    if agg == "1H":
        agg_suffix = "hours"
    else:
        agg_suffix = "days"
    return df, agg_suffix


def draw_prediction_plot(results, agg_suffix):

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=results.index,
            y=results[target_variable],
            mode="lines",
            name="Actual",
            line=dict(color="grey"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=results.index,
            y=results["prophet"],
            mode="lines",
            name="Prophet",
            line=dict(color=col_red),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=results.index,
            y=results["sarimax"],
            mode="lines",
            name="SARIMAX",
            line=dict(color=col_blue),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=results.index,
            y=results["holtwinters"],
            mode="lines",
            name="Holt-Winters",
            line=dict(color=col_magenta),
        )
    )
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Watt Hours",
        margin=go.layout.Margin(
            l=margin_l, r=margin_r, b=margin_b, t=margin_t, pad=margin_pad
        ),
    )
    fig.layout.template = template
    fig.write_image(
        os.path.join(fig_path, img_chart_name + "_" + agg_suffix + ".png"),
        width=img_width,
        height=img_height,
        scale=img_scale,
    )
    fig.show()


def draw_metrics_table(results, agg_suffix):
    index_type = ["Insample"] + [
        "Test {}".format(test_number) for test_number in range(1, tests + 1)
    ]
    index_type = index_type * 3
    index_metric = ["MSE"] * (tests + 1) + ["MAE"] * (tests + 1) + ["R2"] * (tests + 1)
    index = pd.MultiIndex.from_arrays(
        [index_type, index_metric], names=("Type", "Metric")
    )
    columns = ["Prophet", "SARIMAX", "Holt-Winters"]
    result_metrics = pd.DataFrame(index=index, columns=columns)

    models = ["prophet", "sarimax", "holtwinters"]
    for model, model_column in zip(models, columns):
        metrics = []
        for metric in [mean_squared_error, mean_absolute_error, r2_score]:
            for type in results["type"].unique():
                metrics.append(
                    round(
                        metric(
                            results.loc[results["type"] == type, target_variable],
                            results.loc[results["type"] == type, model],
                        )
                    )
                )
        result_metrics[model_column] = metrics

    result_metrics.reset_index(inplace=True)

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(values=list(result_metrics.columns), align="left"),
                cells=dict(
                    values=[
                        result_metrics[column].to_list()
                        for column in result_metrics.columns
                    ],
                    align="left",
                ),
            )
        ]
    )
    fig.update_layout(margin=go.layout.Margin(l=0, r=0, b=0, t=0, pad=10))
    fig.layout.template = template
    fig.write_image(
        os.path.join(fig_path, img_table_name + "_" + agg_suffix + ".png"),
        width=img_width,
        height=(img_height - 50) / 2,
        scale=img_scale,
    )
    fig.show()


def model_and_plot(
    data,
    agg_levels,
    prophet_freq,
    sarimax_order,
    sarimax_seasonal_order,
    holtwinters_seasonal_periods,
):

    # getting the right data aggregation
    data, agg_suffix = aggregate_df(data, agg_levels[0])

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

    results = prophet_cv(results=results, freq=prophet_freq)
    results = sarimax_cv(
        results=results, order=sarimax_order, seasonal_order=sarimax_seasonal_order
    )
    results = holtwinters_cv(
        results=results, seasonal_periods=holtwinters_seasonal_periods
    )

    for agg in agg_levels:

        results, agg_suffix = aggregate_df(results, agg)
        draw_prediction_plot(results=results, agg_suffix=agg_suffix)
        draw_metrics_table(results=results, agg_suffix=agg_suffix)


#%%

model_and_plot(
    data,
    agg_levels=["1H", "1D"],
    prophet_freq="H",
    sarimax_order=(2, 1, 1),
    sarimax_seasonal_order=(2, 0, 1, 24),
    holtwinters_seasonal_periods=24,
)


#%%


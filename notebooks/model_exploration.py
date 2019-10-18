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
data = data["2007-01-01 00:00:00":"2009-12-31 23:59:00"]

#%%

# GLOBALS

target_variable = "Global_active_power"
# 24 * 30 = 30 days = 1 month
hourmodel = dict(
    data=data,
    test_size=24 * 30,
    tests=3,
    agg_levels=["1H", "1D", "30D"],
    fig_path=os.path.join("reports", "figures"),
    img_timeseries_name="model_comparison_timeseries_hourmodel",
    img_table_name="model_comparison_table_hourmodel",
    prophet_freq="H",
    sarimax_order=(2, 1, 1),
    sarimax_seasonal_order=(2, 0, 1, 24),
    holtwinters_seasonal_periods=24,
)

daymodel = dict(
    data=data,
    test_size=30,
    tests=3,
    agg_levels=["1D", "30D"],
    fig_path=os.path.join("reports", "figures"),
    img_timeseries_name="model_comparison_timeseries_daymodel",
    img_table_name="model_comparison_table_daymodel",
    prophet_freq="D",
    sarimax_order=(2, 1, 1),
    sarimax_seasonal_order=(1, 0, 1, 7),
    holtwinters_seasonal_periods=7,
)

models = [hourmodel, daymodel]

#%%


def prophet_cv(data, tests, test_size, results, freq):

    for test_number in range(1, tests + 1):

        test_split = len(data) - test_size * (tests - test_number + 1)

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


def sarimax_cv(data, tests, test_size, results, order, seasonal_order):

    for test_number in range(1, tests + 1):

        test_split = len(data) - test_size * (tests - test_number + 1)

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


def holtwinters_cv(data, tests, test_size, results, seasonal_periods):

    for test_number in range(1, tests + 1):

        test_split = len(data) - test_size * (tests - test_number + 1)

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


def average_cv(data, tests, test_size, results):

    # for average we use only the data from the previous test period
    for test_number in range(1, tests + 1):

        split_start = len(data) - test_size * (tests - test_number + 2)
        split_end = len(data) - test_size * (tests - test_number + 1)

        df_train = data[split_start:split_end][[target_variable]]

        mean = df_train[target_variable].mean()

        # for the first test take also the insample predictions
        if test_number == 1:
            predictions = [mean] * test_size * 2
        else:
            predictions = predictions + [mean] * test_size

    results["average"] = predictions

    return results


def aggregate_df(df, agg, cat_column=None):
    if cat_column:
        df = df.groupby([pd.Grouper(freq=agg), cat_column]).mean()
        df.reset_index(level=1, inplace=True)
    else:
        df = df.resample(agg).mean()

    if agg == "1H":
        agg_suffix = "hours"
    elif agg == "1D":
        agg_suffix = "days"
    else:
        agg_suffix = "months"
    return df, agg_suffix


def draw_prediction_plot(results, agg_suffix, fig_path, img_timeseries_name):

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=results.index,
            y=results[target_variable],
            mode="lines",
            name="Actual",
            line=dict(color="grey", width=1),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=results.index,
            y=results["prophet"],
            mode="lines",
            name="Prophet",
            line=dict(color=col_red, width=1),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=results.index,
            y=results["sarimax"],
            mode="lines",
            name="SARIMAX",
            line=dict(color=col_blue, width=1),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=results.index,
            y=results["holtwinters"],
            mode="lines",
            name="Holt-Winters",
            line=dict(color=col_magenta, width=1),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=results.index,
            y=results["average"],
            mode="lines",
            name="Average",
            line=dict(color="orange", width=1),
        )
    )
    y_max = results.select_dtypes(include=["float"]).values.max() * 1.1
    vlines = []
    for i, start_point in enumerate(
        results.reset_index().groupby("type")["Date_Time"].min().to_list()
    ):
        if i != 0:
            vlines.append(
                {
                    "type": "line",
                    "xref": "x",
                    "yref": "y",
                    "x0": start_point,
                    "y0": 0,
                    "x1": start_point,
                    "y1": y_max,
                    "line": dict(color="red", width=1),
                }
            )

    fig.update_layout(shapes=vlines)

    fig.update_layout(
        yaxis=dict(rangemode="tozero"),
        xaxis_title="",
        yaxis_title="Watt Hours",
        margin=go.layout.Margin(
            l=margin_l, r=margin_r, b=margin_b, t=margin_t, pad=margin_pad
        ),
    )
    fig.layout.template = template
    fig.write_image(
        os.path.join(fig_path, img_timeseries_name + "_" + agg_suffix + ".png"),
        width=img_width,
        height=img_height,
        scale=img_scale,
    )
    fig.show()


def draw_metrics_table(tests, results, agg_suffix, fig_path, img_table_name):
    index_type = ["Insample"] + [
        "Test {}".format(test_number) for test_number in range(1, tests + 1)
    ]
    index_type = index_type * 2
    index_metric = ["MSE"] * (tests + 1) + ["MAE"] * (tests + 1)
    index = pd.MultiIndex.from_arrays(
        [index_type, index_metric], names=("Type", "Metric")
    )
    columns = ["Prophet", "SARIMAX", "Holt-Winters", "Average"]
    result_metrics = pd.DataFrame(index=index, columns=columns)

    models = ["prophet", "sarimax", "holtwinters", "average"]
    for model, model_column in zip(models, columns):
        metrics = []
        for metric in [mean_squared_error, mean_absolute_error]:
            for type in results["type"].unique():
                metrics.append(
                    round(
                        metric(
                            results.loc[results["type"] == type, target_variable],
                            results.loc[results["type"] == type, model],
                        ),
                        2,
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
        height=(img_height - 60) / 2,
        scale=img_scale,
    )
    fig.show()


def model_and_plot(
    data,
    agg_levels,
    test_size,
    tests,
    prophet_freq,
    sarimax_order,
    sarimax_seasonal_order,
    holtwinters_seasonal_periods,
    fig_path,
    img_timeseries_name,
    img_table_name,
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
        for test_number in range(1, tests + 1)
        for _ in range(test_size)
    ]

    results = prophet_cv(
        data=data, tests=tests, test_size=test_size, results=results, freq=prophet_freq
    )

    results = sarimax_cv(
        data=data,
        tests=tests,
        test_size=test_size,
        results=results,
        order=sarimax_order,
        seasonal_order=sarimax_seasonal_order,
    )

    results = holtwinters_cv(
        data=data,
        tests=tests,
        test_size=test_size,
        results=results,
        seasonal_periods=holtwinters_seasonal_periods,
    )

    results = average_cv(data=data, tests=tests, test_size=test_size, results=results)

    for agg in agg_levels:

        results_agg, agg_suffix = aggregate_df(results, agg, cat_column="type")

        draw_prediction_plot(
            results=results_agg,
            agg_suffix=agg_suffix,
            fig_path=fig_path,
            img_timeseries_name=img_timeseries_name,
        )
        draw_metrics_table(
            results=results_agg,
            tests=tests,
            agg_suffix=agg_suffix,
            fig_path=fig_path,
            img_table_name=img_table_name,
        )


#%%

for model in models:
    model_and_plot(**model)

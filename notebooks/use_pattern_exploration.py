#%%

import copy
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from dotenv import find_dotenv, load_dotenv
from IPython.core.interactiveshell import InteractiveShell
import plotly.graph_objects as go

from src.data.load_data import load_processed_data

#%% Setting styles

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

#%% Loading in the data

data_path = os.path.join("data", "raw", "submeters.csv")
data = pd.read_csv(data_path, parse_dates=[["Date", "Time"]])
data.set_index("Date_Time", inplace=True)

# There are no missing values but it just because the index is not fully continuous.
# This is does not matter in the analyses that we will be doing here
data.dtypes
data.isnull().sum()
data.describe()
data.head()
data.shape

# calculating unmeasured power use from total power use and coverting it to watt hours.
# (See dataset_info.txt for conversions details).
data["Unmeasured"] = (
    data["Global_active_power"]
    .multiply(1000)
    .divide(60)
    .subtract(data["Sub_metering_1"])
    .subtract(data["Sub_metering_2"])
    .subtract(data["Sub_metering_3"])
)

data.drop(
    columns=[
        "id",
        "Global_active_power",
        "Voltage",
        "Global_reactive_power",
        "Global_intensity",
        "dataset",
    ],
    inplace=True,
)

data.columns = ["Kitchen", "Laundry Room", "Heating", "Unmeasured"]
# 1 min timeframe is too tight for even one day visualizations
data = data.resample("5Min").mean()

#%% Overall trends

img_name = "energy_use_overall_trends.png"
data_trend = data.resample("1M").mean()

#%%

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=data_trend.index.date,
        y=data_trend["Kitchen"],
        mode="lines",
        name="Kitchen",
        line=dict(color=col_red),
    )
)
fig.add_trace(
    go.Scatter(
        x=data_trend.index.date,
        y=data_trend["Laundry Room"],
        mode="lines",
        name="Laundry Room",
        line=dict(color=col_blue),
    )
)
fig.add_trace(
    go.Scatter(
        x=data_trend.index.date,
        y=data_trend["Heating"],
        mode="lines",
        name="Heating",
        line=dict(color=col_magenta),
    )
)
fig.add_trace(
    go.Scatter(
        x=data_trend.index.date,
        y=data_trend["Unmeasured"],
        mode="lines",
        name="Unmeasured",
        line=dict(color="grey"),
    )
)
fig.update_layout(
    yaxis=dict(zeroline=True),
    xaxis_title="",
    yaxis_title="Watt Hours",
    margin=go.layout.Margin(
        l=margin_l, r=margin_r, b=margin_b, t=margin_t, pad=margin_pad
    ),
    annotations=[
        go.layout.Annotation(
            x="2007-01-10",
            y=21,
            xref="x",
            yref="y",
            text="Very high unmeasured use",
            showarrow=True,
            arrowhead=1,
            arrowsize=2,
            ax=120,
            ay=10,
        ),
        go.layout.Annotation(
            x="2008-08-30",
            y=3.5,
            xref="x",
            yref="y",
            text="Travelling?",
            showarrow=True,
            arrowhead=1,
            arrowsize=2,
            ax=-10,
            ay=-180,
        ),
    ],
)
fig.layout.template = template
fig.write_image(
    os.path.join(fig_path, img_name),
    width=img_width,
    height=img_height,
    scale=img_scale,
)
fig.show()

#%% hourly, weekly and monthly trends

data_time = data.groupby(data.index.time).mean().reset_index()
data_time.rename(columns={"index": "Time"}, inplace=True)

data_weekday = data.groupby(data.index.weekday_name).mean().reset_index()
data_weekday.rename(columns={"Date_Time": "Weekday"}, inplace=True)
weekday_order = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]
data_weekday["Weekday"] = pd.Categorical(data_weekday.Weekday, weekday_order)
data_weekday.sort_values(by=["Weekday"], inplace=True)

data_month = data.groupby(data.index.month_name(locale=None)).mean().reset_index()
data_month.rename(columns={"Date_Time": "Month"}, inplace=True)
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
data_month["Month"] = pd.Categorical(data_month.Month, month_order)
data_month.sort_values(by=["Month"], inplace=True)

#%%

img_names = [
    "daily_seasonality.png",
    "weekly_seasonality.png",
    "monthly_seasonality.png",
]
datasets = [data_time, data_weekday, data_month]
xaxis_list = ["Time", "Weekday", "Month"]

for dataset, xaxis, img_name in zip(datasets, xaxis_list, img_names):

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dataset[xaxis],
            y=dataset["Kitchen"],
            mode="lines",
            name="Kitchen",
            line=dict(color=col_red),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dataset[xaxis],
            y=dataset["Laundry Room"],
            mode="lines",
            name="Laundry Room",
            line=dict(color=col_blue),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dataset[xaxis],
            y=dataset["Heating"],
            mode="lines",
            name="Heating",
            line=dict(color=col_magenta),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dataset[xaxis],
            y=dataset["Unmeasured"],
            mode="lines",
            name="Unmeasured",
            line=dict(color="grey"),
        )
    )
    fig.update_layout(
        yaxis=dict(zeroline=True, range=[0, max(dataset["Unmeasured"]) + 1]),
        xaxis_title="",
        yaxis_title="Watt Hours",
        margin=go.layout.Margin(
            l=margin_l, r=margin_r, b=margin_b, t=margin_t, pad=margin_pad
        ),
    )
    if xaxis == "Time":
        fig.update_layout(xaxis=dict(tickmode="linear", tick0=0, dtick=24))
    fig.layout.template = template
    if xaxis == "Time":
        fig.update_layout(
            annotations=[
                go.layout.Annotation(
                    x="03:00:00",
                    y=2.4,
                    xref="x",
                    yref="y",
                    text="Not much heating or air conditioning during the night",
                    showarrow=True,
                    arrowhead=1,
                    arrowsize=2,
                    ax=80,
                    ay=-130,
                ),
                go.layout.Annotation(
                    x="06:45:00",
                    y=15,
                    xref="x",
                    yref="y",
                    text="Unknown peak",
                    showarrow=True,
                    arrowhead=1,
                    arrowsize=2,
                    ax=-120,
                    ay=-20,
                ),
                go.layout.Annotation(
                    x="19:00:00",
                    y=18,
                    xref="x",
                    yref="y",
                    text="Evening entertainment electronics",
                    showarrow=True,
                    arrowhead=1,
                    arrowsize=2,
                    ax=-180,
                    ay=-10,
                ),
                go.layout.Annotation(
                    x="08:10:00",
                    y=2.2,
                    xref="x",
                    yref="y",
                    text="Breakfast",
                    showarrow=True,
                    arrowhead=1,
                    arrowsize=2,
                    ax=-10,
                    ay=-50,
                ),
            ]
        )
    elif xaxis == "Weekday":
        fig.update_layout(
            annotations=[
                go.layout.Annotation(
                    x="Saturday",
                    y=10.1,
                    xref="x",
                    yref="y",
                    text="Increased entertainment electronics use during weekend?",
                    showarrow=True,
                    arrowhead=7,
                    arrowsize=2,
                    ax=-250,
                    ay=-10,
                )
            ]
        )
    else:
        fig.update_layout(
            annotations=[
                go.layout.Annotation(
                    x="August",
                    y=5.3,
                    xref="x",
                    yref="y",
                    text="Heating only dips in July and August",
                    showarrow=True,
                    arrowhead=1,
                    arrowsize=2,
                    ax=-50,
                    ay=-100,
                )
            ]
        )
    fig.write_image(
        os.path.join(fig_path, img_name),
        width=img_width,
        height=img_height,
        scale=img_scale,
    )
    fig.show()


#%% Typical Day

data_20080708 = data["2008-07-08"].copy()
data_20091224 = data["2009-12-24"].copy()

data_20080708["Time"] = data_20080708.index.time
data_20091224["Time"] = data_20091224.index.time

#%%

img_names = ["20080607.png", "20091224.png"]
datasets = [data_20080708, data_20091224]

for dataset, img_name in zip(datasets, img_names):

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dataset.Time,
            y=dataset["Kitchen"],
            mode="lines",
            name="Kitchen",
            line=dict(color=col_red),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dataset.Time,
            y=dataset["Laundry Room"],
            mode="lines",
            name="Laundry Room",
            line=dict(color=col_blue),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dataset.Time,
            y=dataset["Heating"],
            mode="lines",
            name="Heating",
            line=dict(color=col_magenta),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dataset.Time,
            y=dataset["Unmeasured"],
            mode="lines",
            name="Unmeasured",
            line=dict(color="grey"),
        )
    )
    fig.update_layout(
        xaxis_title="",
        yaxis_title="Watt Hours",
        xaxis=dict(tickmode="linear", tick0=0, dtick=24),
        yaxis=dict(range=(0, 60)),
        margin=go.layout.Margin(
            l=margin_l, r=margin_r, b=margin_b, t=margin_t, pad=margin_pad
        ),
    )
    fig.layout.template = template
    if img_name == "20080607.png":
        fig.update_layout(
            annotations=[
                go.layout.Annotation(
                    x="02:00:00",
                    y=12,
                    xref="x",
                    yref="y",
                    text="Air conditioning spikes",
                    showarrow=True,
                    arrowhead=1,
                    arrowsize=2,
                    ax=10,
                    ay=-80,
                ),
                go.layout.Annotation(
                    x="07:55:00",
                    y=2,
                    xref="x",
                    yref="y",
                    text="Fridge in the laundry room",
                    showarrow=True,
                    arrowhead=1,
                    arrowsize=2,
                    ax=120,
                    ay=-250,
                ),
                go.layout.Annotation(
                    x="07:20:00",
                    y=19,
                    xref="x",
                    yref="y",
                    text="Morning shower",
                    showarrow=True,
                    arrowhead=1,
                    arrowsize=2,
                    ax=-120,
                    ay=-120,
                ),
            ]
        )
    else:
        fig.update_layout(
            annotations=[
                go.layout.Annotation(
                    x="02:45:00",
                    y=2.1,
                    xref="x",
                    yref="y",
                    text="Fridge is on much less during the winter",
                    showarrow=True,
                    arrowhead=1,
                    arrowsize=2,
                    ax=50,
                    ay=-50,
                ),
                go.layout.Annotation(
                    x="15:00:00",
                    y=43,
                    xref="x",
                    yref="y",
                    text="Christmas cleaning",
                    showarrow=True,
                    arrowhead=1,
                    arrowsize=2,
                    ax=-150,
                    ay=-50,
                ),
                go.layout.Annotation(
                    x="19:00:00",
                    y=43,
                    xref="x",
                    yref="y",
                    text="Cooking the Christman Dinner",
                    showarrow=True,
                    arrowhead=1,
                    arrowsize=2,
                    ax=-140,
                    ay=-110,
                ),
                go.layout.Annotation(
                    x="08:00:00",
                    y=20,
                    xref="x",
                    yref="y",
                    text="Morning cartoons?",
                    showarrow=True,
                    arrowhead=1,
                    arrowsize=2,
                    ax=-30,
                    ay=-90,
                ),
            ]
        )
    fig.write_image(
        os.path.join(fig_path, img_name),
        width=img_width,
        height=img_height,
        scale=img_scale,
    )
    fig.show()

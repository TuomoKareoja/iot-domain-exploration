#%%

import copy
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import sklearn
from dotenv import find_dotenv, load_dotenv
from IPython.core.interactiveshell import InteractiveShell

from src.data.load_data import load_processed_data

# Setting styles
InteractiveShell.ast_node_interactivity = "all"
sns.set(style="whitegrid", color_codes=True)

#%% Loading in the data
data_raw_path = os.path.join("data", "raw", "submeters.csv")
data_raw = pd.read_csv(data_raw_path, parse_dates=[["Date", "Time"]])
data_raw.set_index("Date_Time", inplace=True)

# There are no missing values but it just because the index is not fully continuous.
# This is does not matter in the analyses that we will be doing here
data_raw.dtypes
data_raw.isnull().sum()
data_raw.describe()
data_raw.head()
data_raw.shape

data_raw["Unmeasured"] = (
    data_raw["Global_active_power"]
    .multiply(1000)
    .divide(60)
    .subtract(data_raw["Sub_metering_1"])
    .subtract(data_raw["Sub_metering_2"])
    .subtract(data_raw["Sub_metering_3"])
)

data_raw.drop(
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

data_raw.columns = ["Submeter 1", "Submeter 2", "Submeter 3", "Unmeasured"]


#%% Typical Day

# Use minute granularity

data_20080407 = data_raw["2008-04-07"].copy()
data_20080708 = data_raw["2008-07-08"].copy()
data_20090108 = data_raw["2009-01-08"].copy()
data_20091224 = data_raw["2009-12-24"].copy()

data_20080407 = data_20080407.resample("5Min").mean()
data_20080708 = data_20080708.resample("5Min").mean()
data_20090108 = data_20090108.resample("5Min").mean()
data_20091224 = data_20091224.resample("5Min").mean()

data_20080407["Time"] = data_20080407.index.time
data_20080708["Time"] = data_20080708.index.time
data_20090108["Time"] = data_20090108.index.time
data_20091224["Time"] = data_20091224.index.time

data_20080407_melt = data_20080407.melt(
    id_vars=["Time"],
    value_vars=["Submeter 1", "Submeter 2", "Submeter 3", "Unmeasured"],
    value_name="Watt Hours",
    var_name="Submeter",
)
data_20080708_melt = data_20080708.melt(
    id_vars=["Time"],
    value_vars=["Submeter 1", "Submeter 2", "Submeter 3", "Unmeasured"],
    value_name="Watt Hours",
    var_name="Submeter",
)
data_20090108_melt = data_20090108.melt(
    id_vars=["Time"],
    value_vars=["Submeter 1", "Submeter 2", "Submeter 3", "Unmeasured"],
    value_name="Watt Hours",
    var_name="Submeter",
)
data_20091224_melt = data_20091224.melt(
    id_vars=["Time"],
    value_vars=["Submeter 1", "Submeter 2", "Submeter 3", "Unmeasured"],
    value_name="Watt Hours",
    var_name="Submeter",
)

#%%

dataset_melt_list = [
    data_20080407_melt,
    data_20080708_melt,
    data_20090108_melt,
    data_20091224_melt,
]

titles = [
    "2008 April Monday",
    "2008 July Tuesday",
    "2009 January Thursday",
    "2009 Christmas Eve",
]

for dataset, title in zip(dataset_melt_list, titles):
    fig = px.line(
        dataset,
        x="Time",
        y="Watt Hours",
        color="Submeter",
        color_discrete_sequence=["green", "blue", "red", "grey"],
        title=title,
    )
    fig.show()

#%% hourly, weekly and monthly trends

# TODO show distributions and lines if legible
data = data_raw.groupby(data_raw.index.weekday_name).mean().reset_index()
data.rename(columns={"Date_Time": "Weekday"}, inplace=True)

data_melt = data.melt(
    id_vars=["Weekday"],
    value_vars=["Submeter 1", "Submeter 2", "Submeter 3", "Unmeasured"],
    value_name="Watt Hours",
    var_name="Submeter",
)

fig = px.line(
    data_melt,
    x="Weekday",
    y="Watt Hours",
    color="Submeter",
    color_discrete_sequence=["green", "blue", "red", "grey"],
)

fig.show()


#%% Extreme values

# TODO
# Where are the extreme values and that submeter is causing them?
# When is the energy use very low and could we see when they are on
# vacation

#%% Unmeasured use patterns

#

#%%

import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import os
import copy
from IPython.core.interactiveshell import InteractiveShell

# Setting styles
InteractiveShell.ast_node_interactivity = "all"
sns.set(style="whitegrid", color_codes=True)

#%%

# load data
weather_path = os.path.join("data", "external", "weather.csv")
data = pd.read_csv(weather_path, index_col=["time"], parse_dates=["time"])
data.head()

#%%

data.isnull().mean().plot.bar()
data.isnull().sum()

# Precipitation accumulation and type have most values missing

#%%

# is precipitation type missing when there is rain?
data.query("precipIntensity > 0").precipType.isnull().sum()

# no missing values when actually raining

#%%


def plot_missing_period(data, variable):
    # Finding the biggest holes in the dataseries

    # new dataframe with only id
    data_missing = data[[variable]].copy()

    # check that first number is not missing
    data_missing.iloc[0, :]

    # add new colum that is 1 when id not null
    data_missing["not_missing"] = np.where(data_missing[variable].isnull(), 0, 1)

    # shift values of new column on forward
    data_missing["not_missing_shifted"] = data_missing.not_missing.shift(1)

    # keep only values where id is missing
    data_missing = data_missing[data_missing[variable].isnull()]

    # drop other columns that shifted marker
    data_missing.drop(columns=[variable, "not_missing", "not_missing"], inplace=True)

    # calculate rolling sum to create id fro
    data_missing["missing_period_id"] = data_missing.not_missing_shifted.cumsum()

    # drop unnecessary column
    data_missing.drop(columns=["not_missing_shifted"], inplace=True)

    # Name the episode by with the start and end time
    data_missing["datetime"] = data_missing.index
    data_missing = data_missing.groupby(["missing_period_id"]).agg(
        ["min", "max", "count"]
    )
    data_missing.columns = data_missing.columns.get_level_values(0)
    data_missing.columns = ["start", "end", "length_hours"]
    data_missing["start"] = data_missing.start.apply(
        lambda x: x.strftime("%Y-%m-%d %H:%M")
    )
    data_missing["end"] = data_missing.end.apply(lambda x: x.strftime("%Y-%m-%d %H:%M"))
    data_missing["period"] = data_missing.start + " to " + data_missing.end
    data_missing.reset_index(level=0, inplace=True)
    data_missing.drop(columns=["start", "end", "missing_period_id"], inplace=True)
    data_missing.set_index(data_missing.period, inplace=True)
    data_missing.drop(columns=["period"], inplace=True)

    # plot missing periods
    data_missing.plot.bar(figsize=(12, 9))


#%%

variables_with_missing_values = [
    "cloudCover",
    "precipIntensity",
    "windGust",
    "precipAccumulation",
]
for variable in variables_with_missing_values:
    plot_missing_period(data, variable)

# periods of missing data are small for cloud cover and precipitation and can be
# corrected with interpolation

# wind gust is a more difficult case as some of the gaps in the information are quite
# large. Probably this variable is of not much information value as we are really just
# trying to find if the weather is not so nice and people will stay inside
# We could first just drop the variable and check later if we have time if it is
# of any extra value

# precip accumulation is hopeless

#%%

# Dropping the uninteresting or broken stuff
data.drop(columns=["precipAccumulation", "windGust"], inplace=True)

#%%

# Checking distributions of the variables

for column in data.columns:
    data[column].hist(bins=100)
    plt.title(column)
    plt.show()

# apparent temperature has a dip in value frequency around freezing point. This
# is probably a some real physical phenomena

# cloudCover has certain values that seem over represented. Zero cloud cover makes sense
# as the mode, but there seems to be some rounding mistakes after 0.3 cloud cover.
# The changes in frequency after 0.7 are probably also a real phenomena related to storms
# but the spike around 0.75 is suspicious

#%%

# We need to make convert the Fahrenheits to Celsius as we are working for an european
# company

data["apparentTemperature_c"] = data["apparentTemperature"].subtract(32).divide(1.8)
data["temperature_c"] = data["temperature"].subtract(32).divide(1.8)
data["apparentTemperature_c"].resample("D").mean().plot()
plt.show()
data["temperature_c"].resample("D").mean().plot()
plt.show()

# looks ok. Apparent temperature differences are bigger than actual ones

#%%

data.drop(columns=["apparentTemperature", "temperature"], inplace=True)

#%%

# heatmaps
corr_matrix = data.select_dtypes(include=["float64"]).corr()

sns.set(rc={"figure.figsize": (11.7, 8.27)})
sns.heatmap(corr_matrix, annot=True, cmap="Reds")

# temperatures and dew point are highly correlated. This might pose problems in the analysis.
# Maybe we should just keep the apparent temperature because it includes the combined information
# of dew point and temperature?


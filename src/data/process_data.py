# -*- coding: utf-8 -*-
import calendar
import os

import numpy as np
import pandas as pd
from fbprophet import Prophet
from pandas.tseries.offsets import MonthEnd
from datetime import datetime, timedelta


def add_correct_index_and_prune(
    inpath=os.path.join("data", "raw", "submeters.csv"),
    outpath=os.path.join("data", "interim", "data_indexed.csv"),
):
    """Adds correct indexing to the raw data

    Keyword Arguments:
        inpath {path} -- [path for incoming data] (default: {os.path.join("data", "raw", "submeters.csv")})
        outpath {path} -- [path for where to save data] (default: {os.path.join("data", "interim", "data_indexed.csv")})
    """

    data = pd.read_csv(inpath, parse_dates=[["Date", "Time"]])

    data.set_index("Date_Time", inplace=True)

    # broadening the index
    data = data.asfreq("min")

    # dropping columns not necessary for the analysis
    data.drop(
        columns=[
            "id",
            "dataset",
            "Global_reactive_power",
            "Global_intensity",
            "Voltage",
        ],
        inplace=True,
    )

    # resampling the data to one hour bins to make processing faster
    # for this analysis we lose no meaning details for doing this
    data = data.resample("H").mean()

    data.to_csv(outpath)


def convert_units_and_add_unmeasured_consumption(
    inpath=os.path.join("data", "interim", "data_indexed.csv"),
    outpath=os.path.join("data", "interim", "data_indexed_converted.csv"),
):
    """Changes global active power units (kilowatt-minutes) to match those of submeters (watt-hours) and adds columns for consumption not measured by submeters

    Keyword Arguments:
        inpath {path} -- [path for incoming data] (default: {os.path.join("data", "interim", "data_indexed.csv")})
        outpath {path} -- [path for where to save data] (default: {os.path.join("data", "interim", "data_indexed_converted.csv")})
    """
    data = pd.read_csv(inpath, parse_dates=["Date_Time"], index_col="Date_Time")

    # changing global active power units from kilowat minutes to kilowat hours
    data["Global_active_power"] = data["Global_active_power"].multiply(1000).divide(60)

    # calculating the power consumption not measured by submeters
    data["unmeasured"] = (
        data["Global_active_power"]
        .subtract(data["Sub_metering_1"])
        .subtract(data["Sub_metering_2"])
        .subtract(data["Sub_metering_3"])
    )

    data.to_csv(outpath)


def add_time_information(
    inpath=os.path.join("data", "interim", "data_indexed_converted.csv"),
    outpath=os.path.join("data", "interim", "data_indexed_converted_timed.csv"),
):
    """Adds columns for year, month, weekday, hour and holiday

    Keyword Arguments:
        inpath {path} -- [path for incoming data] (default: {os.path.join("data", "interim", "data_indexed_converted.csv")})
        outpath {path} -- [path for where to save data] (default: {os.path.join("data", "interim", "data_indexed_converted_timed.csv")})
    """

    import holidays

    fra_holidays = holidays.CountryHoliday("FRA")

    data = pd.read_csv(inpath, parse_dates=["Date_Time"], index_col="Date_Time")

    data["year"] = data.index.year
    data["month"] = data.index.month_name()
    data["weekday"] = data.index.weekday_name
    data["hour"] = data.index.hour

    data["date"] = data.index.date
    data["holiday"] = data["date"].apply(lambda x: x in fra_holidays)
    data.drop(columns=["date"], inplace=True)

    data.to_csv(outpath)


def fill_missing_with_prophet(
    columns_to_fill,
    inpath=os.path.join("data", "interim", "data_indexed_converted_timed.csv"),
    outpath=os.path.join("data", "interim", "data_indexed_converted_timed_filled.csv"),
):
    """Replaces missing values from insample predictions from a Prophet model

    Arguments:
        columns_to_fill {list} -- Columns to fill with a model

    Keyword Arguments:
        inpath {string} -- Path to raw weather data (default: {os.path.join("data", "external", "data_indexed_converted_timed.csv")})
        outpath {string} -- Path for processed weather data (default: {os.path.join("data", "interim", "data_indexed_converted_timed_filled.csv")})
    """
    data = pd.read_csv(inpath, parse_dates=["Date_Time"], index_col="Date_Time")

    for column in columns_to_fill:

        data_prophet = data[column].reset_index(level=0)
        data_prophet.columns = ["ds", "y"]

        m = Prophet()
        m.fit(data_prophet)
        # predicting just the insample
        future = m.make_future_dataframe(periods=0, freq="H")
        forecast = m.predict(future)
        forecast["yhat"] = np.where(forecast["yhat"] < 0, 0, forecast["yhat"])
        forecast.set_index("ds", inplace=True)
        forecast.index.names = ["Date_Time"]
        # we only need the predictions and index
        forecast = forecast[["yhat"]]
        # joining by indices
        data = data.join(forecast)
        # replacing missing with the prediction
        data[column].fillna(data["yhat"], inplace=True)
        # dropping the predictions so that they wont' interfere with futher joins
        data.drop(columns=["yhat"], inplace=True)

    data.to_csv(outpath)


def convert_and_clean_weather_dataset(
    inpath=os.path.join("data", "external", "weather.csv"),
    outpath=os.path.join("data", "interim", "weather_pruned_converted.csv"),
):
    """Processes weather data. Drops unnecessary columns, converts temperature units to celsius, fills in missing values

    Keyword Arguments:
        inpath {string} -- Path to raw weather data (default: {os.path.join("data", "external", "weather.csv")})
        outpath {string} -- Path for processed weather data (default: {os.path.join("data", "interim", "weather_pruned_converted.csv")})
    """
    data = pd.read_csv(inpath, parse_dates=["time"], index_col="time")

    # dropping unnecessary columns
    data.drop(columns=["precipAccumulation", "windGust"], inplace=True)

    # if precipitation type is null = clear
    data["precipType"].fillna("clear", inplace=True)

    # linear interpolation for numeric values with missing values
    for column in ["cloudCover", "precipIntensity"]:
        data[column].interpolate(inplace=True)

    # convertint temperature from Fahrenheit to Celsius
    data["apparentTemperature"] = data["apparentTemperature"].subtract(32).divide(1.8)
    data["temperature"] = data["temperature"].subtract(32).divide(1.8)

    # renaming index to match electricity use data
    data.index.names = ["Date_Time"]

    data.to_csv(outpath)


def combine_datasets(
    inpaths=[
        os.path.join("data", "interim", "data_indexed_converted_timed_filled.csv"),
        os.path.join("data", "interim", "weather_pruned_converted.csv"),
    ],
    outpath=os.path.join("data", "processed", "data_ready.csv"),
):
    """Combines electricity use and weather data
    
    Keyword Arguments:
        inpaths {list} -- list of paths to electricity and weather data (default: {[os.path.join("data", "interim", "data_indexed_converted_timed_filled.csv"),os.path.join("data", "interim", "weather_pruned_converted.csv"),]})
        outpath {string} -- Path for combined dataset (default: {os.path.join("data", "processed", "data_ready.csv")})
    """
    datasets = []
    for path in inpaths:
        df = pd.read_csv(path, index_col=["Date_Time"], parse_dates=["Date_Time"])
        datasets.append(df)

    # concat by columns = full outer join by datetimeindex
    data = pd.concat(datasets, axis=1)

    # dropping rows that don't have a match in electricity data
    data.dropna(subset=["hour"], inplace=True)

    # make year and hour integers
    data[["hour", "year"]] = data[["hour", "year"]].astype(int)

    data.to_csv(outpath)


def make_tableau_dataset(
    inpath=os.path.join("data", "processed", "data_ready.csv"),
    outpath=os.path.join("data", "processed", "data_ready_tableau.csv"),
):
    """Creates a csv file to use in Tableau dashboard
    
    Keyword Arguments:
        inpath {string} -- Path to the last iteration of data (default: {os.path.join("data", "processed", "data_ready.csv")})
        outpath {string} -- Path to output file (default: {os.path.join("data", "processed", "data_ready_tableau.csv")})
    """

    # remove total energy use and all weather information
    df = pd.read_csv(inpath, index_col=["Date_Time"], parse_dates=["Date_Time"])
    df = df[["Sub_metering_1", "Sub_metering_2", "Sub_metering_3", "unmeasured"]]
    columns = ["Kitchen", "Laundry Room", "Heating and Air Conditioning", "Other"]
    df.columns = columns

    last_day = df.index[-1]
    month_end = last_day + MonthEnd(1)
    month_end = month_end.replace(hour=23)
    diff = month_end - last_day
    hours_to_predict = int(diff.total_seconds() / 3600)

    # add predictions for all submeters with a predictions up to full month

    index = pd.date_range(
        last_day + timedelta(hours=1), periods=hours_to_predict, freq="H"
    )
    predictions_df = pd.DataFrame(index=index, columns=columns)
    predictions_df.index.name = "Date_Time"

    for column in columns:

        data_prophet = df[column].reset_index(level=0)
        data_prophet.columns = ["ds", "y"]

        m = Prophet()
        m.fit(data_prophet)
        future = m.make_future_dataframe(periods=hours_to_predict, freq="H")
        forecast = m.predict(future)
        predictions_df[column] = forecast["yhat"][-hours_to_predict:]

    df["prediction"] = False
    predictions_df["prediction"] = True

    df = pd.concat([df, predictions_df])

    # Add boolean column for current month and last month
    df["month"] = np.where(
        (df.index.year == last_day.year) & (df.index.month == last_day.month),
        'Current Month',
        '',
    )
    df["month"] = np.where(
        (
            (df.index.year == last_day.year)
            & (df.index.month == last_day.month - 1)
            & (last_day.month != 1)
        )
        | (
            (df.index.year == last_day.year - 1)
            & (df.index.month == 12)
            & (last_day.month == 1)
        ),
        'Last Month',
        df['month'],
    )

    df.reset_index(level=0, inplace=True)
    df = df.melt(
        id_vars=["Date_Time", "prediction", "month"],
        var_name="measure",
        value_name="Value",
    )

    df.to_csv(outpath, index=False)

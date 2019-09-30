# -*- coding: utf-8 -*-
import calendar
import os

import numpy as np
import pandas as pd


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

    # keeping only full years of data
    data = data["2007":"2010"]

    # dropping columns not necessary for the analysis
    data.drop(
        columns=["id", "dataset", "Global_reactive_power", "Voltage"], inplace=True
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

    # limiting timeseries to only 200-2010
    data = data["2007":"2010"]

    # dropping unnecessary columns
    data.drop(columns=["precipAccumulation", "windGust"], inplace=True)

    # if precipitation type is null = clear
    data["precipType"].fillna("clear", inplace=True)

    # linear interpolation for numeric values with missing values
    for column in ["cloudCover", "precipIntensity", "windGust"]:
        data[column].interpolate(inplace=True)

    # convertint temperature from Fahrenheit to Celsius
    data["apparentTemperature"] = data["apparentTemperature"].subtract(32).divide(1.8)
    data["temperature"] = data["temperature"].subtract(32).divide(1.8)

    # renaming index to match electricity use data
    data.index.names = ["Date_Time"]

    data.to_csv(outpath)


def combine_datasets(
    inpaths=[
        os.path.join("data", "interim", "data_indexed_converted_timed.csv"),
        os.path.join("data", "interim", "weather_pruned_converted.csv"),
    ],
    outpath=os.path.join("data", "processed", "data_ready.csv"),
):
    datasets = []
    for path in inpaths:
        df = pd.read_csv(path)
        datasets.append(df)

    # concat by columns = full outer join by datetimeindex
    data = pd.concat(datasets, axis=1)

    data.to_csv(outpath, index=False)

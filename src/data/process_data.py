# -*- coding: utf-8 -*-
import calendar
import os

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar as holiday_calendar


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

    data.to_csv(outpath)


def convert_units(
    inpath=os.path.join("data", "interim", "data_indexed.csv"),
    outpath=os.path.join("data", "interim", "data_indexed_converted.csv"),
):
    """Changes global active power units (kilowatt-minutes) to match those of submeters (watt-hours)

    Keyword Arguments:
        inpath {path} -- [path for incoming data] (default: {os.path.join("data", "interim", "data_indexed.csv")})
        outpath {path} -- [path for where to save data] (default: {os.path.join("data", "interim", "data_indexed_converted.csv")})
    """
    data = pd.read_csv(inpath, parse_dates=["Date_Time"], index_col="Date_Time")

    # changing global active power units from kilowat minutes to kilowat hours
    data["Global_active_power"] = data["Global_active_power"].multiply(1000).divide(60)

    data.to_csv(outpath)


def add_time_information(
    inpath=os.path.join("data", "interim", "data_indexed_converted.csv"),
    outpath=os.path.join("data", "processed", "data_ready.csv"),
):
    """Adds columns for year, month, weekday, hour and holiday

    Keyword Arguments:
        inpath {path} -- [path for incoming data] (default: {os.path.join("data", "interim", "data_indexed_converted.csv")})
        outpath {path} -- [path for where to save data] (default: {os.path.join("data", "processed", "data_ready.csv")})

    """

    data = pd.read_csv(inpath, parse_dates=["Date_Time"], index_col="Date_Time")

    data["year"] = data.index.year
    data["month"] = data.index.month_name()
    data["weekday"] = data.index.weekday_name
    data["hour"] = data.index.hour

    cal = holiday_calendar()
    holidays = cal.holidays(start=data.index.min(), end=data.index.max())
    data["holiday"] = data.index.isin(holidays)

    data.to_csv(outpath)

# -*- coding: utf-8 -*-
import calendar
import os

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar as holiday_calendar


def add_correct_index(
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
    data.drop(columns=["id"], inplace=True)

    data.to_csv(outpath)


def add_time_information(
    inpath=os.path.join("data", "interim", "data_indexed.csv"),
    outpath=os.path.join("data", "processed", "data_ready.csv"),
):
    """Adds columns for month, weekday, hour and holiday

    Keyword Arguments:
        inpath {path} -- [path for incoming data] (default: {os.path.join("data", "interim", "data_indexed.csv")})
        outpath {path} -- [path for where to save data] (default: {os.path.join("data", "processed", "data_ready.csv")})

    """

    data = pd.read_csv(inpath, parse_dates=["Date_Time"], index_col="Date_Time")

    data["month"] = data.index.month_name()
    data["weekday"] = data.index.weekday_name
    data["hour"] = data.index.hour

    cal = holiday_calendar()
    holidays = cal.holidays(start=data.index.min(), end=data.index.max())
    data["holiday"] = data.index.isin(holidays)

    data.to_csv(outpath)

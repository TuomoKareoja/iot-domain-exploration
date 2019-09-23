# -*- coding: utf-8 -*-
import calendar
import os

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar as holiday_calendar


def load_processed_data(inpath=os.path.join("data", "processed", "data_ready.csv"),):
    """Loads the most processed version of the data

    Keyword Arguments:
        inpath {path} -- [path to the latest version of the processed data] (default: {os.path.join("data", "interim", "data_indexed.csv")})

    Returns:
        [dataframe] -- [The latest version of the processed data]
    """

    data = pd.read_csv(inpath, parse_dates=["Date_Time"], index_col=["Date_Time"])

    return data

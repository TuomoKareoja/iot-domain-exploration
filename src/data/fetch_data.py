# -*- coding: utf-8 -*-
import os

import pandas as pd
from sqlalchemy import create_engine
import pymysql
from datetime import datetime
import requests


def fetch_electricity_data(user, password, host, db, filename):
    """Loads data from database and saves it disk as csv
    
    Arguments:
        user {string} -- database user
        password {string} -- database password
        host {string} -- database host
        db {string} -- database name
        output_filepath {string} -- relative filepath where to save the data
        filename {string} -- Filename of the saved file. Remember .csv!
    """

    connect_url = "mysql+pymysql://" + user + ":" + password + "@" + host + "/" + db
    engine = create_engine(connect_url)
    query = """
        SELECT *, 'data2006' as dataset FROM yr_2006
        UNION ALL
        SELECT *, 'data2007' as dataset FROM yr_2007
        UNION ALL
        SELECT *, 'data2008' as dataset FROM yr_2008
        UNION ALL
        SELECT *, 'data2009' as dataset FROM yr_2009
        UNION ALL
        SELECT *, 'data2010' as dataset FROM yr_2010
    """
    data = pd.read_sql(con=engine, sql=query)
    data.to_csv(os.path.join("data", "raw", filename), index=False)


def fetch_weather_data(date_start, date_end, filename, api_key):
    """Fetches hourly Sceaux weather data for date range from DarkSky API

    Arguments:
        date_start {string} -- first date for weather (format YYYY-MM-DD)
        date_end {string} -- last date for weather (format YYYY-MM-DD)
        filename {string} -- filename for outcome. Remember .csv!
        api_key {string} -- DarkSky API key
    """
    frames = []
    for date in pd.date_range(start=date_start, end=date_end):
        frames.append(get_hourly_data(date=date, api_key=api_key))
    df = pd.concat(frames, sort=True)
    filepath = os.path.join("data", "external", filename)
    df.to_csv(filepath)


def get_hourly_data(date, api_key):
    """Fetches hourly Sceaux weather data for one day from DarkSky API

    Arguments:
        date {datetime} -- Datetime to fetch the data. Uses just the date part
        api_key {string} -- DarkSky API key

    Returns:
        dataframe -- dataframe with weather data and an hourly datetimeindex
    """
    # adding two hours to compensate for french mainland timezone
    unix_timestamp = int((date - datetime(1970, 1, 1)).total_seconds()) + (2 * 60 * 60)
    url = "https://api.darksky.net/forecast/{api_key}/48.7731,2.3012,{unix_timestamp}?exclude=currently,minutely,daily,flags"
    full_url = url.format(unix_timestamp=unix_timestamp, api_key=api_key)
    response = requests.get(
        full_url,
        headers={
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:69.0) Gecko/20100101 Firefox/69.0"
        },
    )
    json_file = response.json()
    df = pd.DataFrame(json_file["hourly"]["data"])

    # converting to datetime index
    df.set_index(pd.to_datetime(df.time, unit="s"), inplace=True)
    # dropping text columns
    df.drop(columns=["time", "summary", "icon"], inplace=True)

    # dropping unnecessary weather columns
    df.drop(
        columns=[
            "precipProbability",
            # dew point and wind gust are probably better measurement of human comfort
            "humidity",
            "windSpeed",
            "pressure",
            "windBearing",
            "uvIndex",
            "visibility",
        ],
        inplace=True,
    )

    return df

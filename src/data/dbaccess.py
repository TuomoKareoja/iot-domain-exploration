# -*- coding: utf-8 -*-
import os

import pandas as pd
from sqlalchemy import create_engine
import pymysql


def fetch_and_save_data(user, password, host, db, filename):
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
    data.to_csv(os.path.join("data", "raw", filename))


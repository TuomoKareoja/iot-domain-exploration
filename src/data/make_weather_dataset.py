# -*- coding: utf-8 -*-
import os

import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from src.data.fetch_data import fetch_weather_data, get_hourly_data


@click.command()
@click.argument("date_start", type=click.Path())
@click.argument("date_end", type=click.Path())
def main(date_start, date_end):
    """ Runs scripts to to fetch external weather data ../extrenal.
    """
    logger = logging.getLogger(__name__)
    logger.info("fetching weather data")
    fetch_weather_data(date_start=date_start, date_end=date_end, api_key=DARKSKYKEY)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    DARKSKYKEY = os.getenv("DARKSKYKEY")

    main()

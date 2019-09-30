# -*- coding: utf-8 -*-
import os

import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from src.data.fetch_data import fetch_electricity_data
from src.data.process_data import (
    add_correct_index_and_prune,
    add_time_information,
    convert_units_and_add_unmeasured_consumption,
    convert_and_clean_weather_dataset,
    combine_datasets,
)


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    # logger.info("fetching data from the database")
    # fetch_electricity_data(
    #     user=DBUSER,
    #     password=DBPASSWORD,
    #     db=DBNAME,
    #     host=DBHOST,
    #     filename="submeters.csv",
    # )

    logger.info("fixing the index and dropping unnecessary columns")
    add_correct_index_and_prune()

    logger.info(
        "converting columns to matching energy units and adding unmeasured consumption"
    )
    convert_units_and_add_unmeasured_consumption()

    logger.info("adding timeperiod columns")
    add_time_information()

    logger.info("cleaning weather data")
    convert_and_clean_weather_dataset()

    logger.info("combining electricty use data to weather information")
    combine_datasets()


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    DBUSER = os.getenv("DBUSER")
    DBPASSWORD = os.getenv("DBPASSWORD")
    DBNAME = os.getenv("DBNAME")
    DBHOST = os.getenv("DBHOST")

    main()

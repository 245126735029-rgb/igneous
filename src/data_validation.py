"""
Simple data-quality checks for NASA FIRMS VIIRS hotspot tables.

These helpers do not confirm fires and do not predict fire risk.
They only look for missing fields and values that cannot be valid
latitude, longitude, or FRP numbers.
"""

import pandas as pd

# Column names taken from the sample VIIRS FIRMS CSV.
REQUIRED_COLUMNS = [
    "latitude",
    "longitude",
    "acq_date",
    "acq_time",
    "satellite",
    "confidence",
    "frp",
]


def validate_required_columns(df):
    """
    Check that every required column is present.

    Returns a list of missing column names.
    An empty list means all required columns exist.
    """
    missing_columns = []
    for column_name in REQUIRED_COLUMNS:
        if column_name not in df.columns:
            missing_columns.append(column_name)
    return missing_columns


def find_missing_values(df):
    """
    Count missing (NaN) values in each required column that exists.

    Returns a pandas Series: column name -> missing-value count.
    """
    columns_to_check = []
    for column_name in REQUIRED_COLUMNS:
        if column_name in df.columns:
            columns_to_check.append(column_name)

    # True where a cell is missing, then count Trues in each column.
    missing_counts = df[columns_to_check].isna().sum()
    return missing_counts


def flag_invalid_latitude(df):
    """
    Flag latitudes that are outside the valid Earth range.

    Returns a boolean Series: True means the latitude is invalid.
    Missing values are not flagged here (see find_missing_values).
    """
    too_low = df["latitude"] < -90
    too_high = df["latitude"] > 90
    return too_low | too_high


def flag_invalid_longitude(df):
    """
    Flag longitudes that are outside the valid Earth range.

    Returns a boolean Series: True means the longitude is invalid.
    Missing values are not flagged here (see find_missing_values).
    """
    too_low = df["longitude"] < -180
    too_high = df["longitude"] > 180
    return too_low | too_high


def flag_negative_frp(df):
    """
    Flag Fire Radiative Power (FRP) values that are negative.

    FRP is a measured energy value; a negative number is not physical.
    Returns a boolean Series: True means FRP is negative.
    """
    return df["frp"] < 0

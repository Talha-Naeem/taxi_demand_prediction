import pandas as pd
from src import config
from src.data import transform_raw_data_into_ts_data

# src/features.py
import numpy as np
import pandas as pd

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build features for taxi demand prediction.
    - Keeps a separate datetime column for train/test splitting.
    - Generates time-based and lag features efficiently.
    """
    # Ensure pickup_ts is datetime
    if not pd.api.types.is_datetime64_any_dtype(df["pickup_ts"]):
        df["pickup_ts"] = pd.to_datetime(df["pickup_ts"], utc=True, errors="coerce")

    # Keep a copy for train/test splitting
    df["pickup_ts_split"] = df["pickup_ts"]

    # Time-based features
    df["hour_of_day"] = df["pickup_ts"].dt.hour
    df["day_of_week"] = df["pickup_ts"].dt.dayofweek
    df["month"] = df["pickup_ts"].dt.month

    # Efficient lag features
    lag_features = [
        df.groupby("pickup_location_id")["rides"].shift(lag).rename(f"lag_{lag}")
        for lag in range(1, config.N_FEATURES + 1)
    ]
    lag_df = pd.concat(lag_features, axis=1)
    df = pd.concat([df, lag_df], axis=1)

    # Target column
    df["target_rides_next_hour"] = df.groupby("pickup_location_id")["rides"].shift(-1)

    # Drop rows with NaNs created by shifting
    df = df.dropna().reset_index(drop=True)

    return df

def build_lag_features(df: pd.DataFrame, lags: int = 653) -> pd.DataFrame:
    """
    Add lag features for demand prediction.
    Assumes df has ['pickup_ts', 'rides'].
    """
    df = df.sort_values("pickup_ts").copy()
    for lag in range(1, lags + 1):
        df[f"lag_{lag}"] = df["rides"].shift(lag)

    # drop rows with NaNs introduced by shifting
    df = df.dropna().reset_index(drop=True)
    return df

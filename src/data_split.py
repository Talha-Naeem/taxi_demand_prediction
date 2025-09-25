from datetime import datetime
from typing import Tuple

import pandas as pd


def train_test_split(df: pd.DataFrame, cutoff_date: pd.Timestamp, target_column_name: str):
    """
    Time-based train/test split using pickup_ts_split column.
    """
    # Split by datetime
    # cutoff_date = features_df["pickup_ts_split"].max() - pd.Timedelta(days=7)

    
    train_data = df[df.pickup_ts_split < cutoff_date].reset_index(drop=True)
    test_data  = df[df.pickup_ts_split >= cutoff_date].reset_index(drop=True)

    X_train = train_data.drop(columns=[target_column_name, "pickup_ts_split"])
    y_train = train_data[target_column_name]

    X_test  = test_data.drop(columns=[target_column_name, "pickup_ts_split"])
    y_test  = test_data[target_column_name]

    # Convert any remaining datetime columns to numeric for LightGBM
    for df_ in [X_train, X_test]:
        for col in df_.columns:
            if pd.api.types.is_datetime64_any_dtype(df_[col]):
                df_[col] = df_[col].astype("int64") // 10**9  # seconds since epoch

    return X_train, y_train, X_test, y_test

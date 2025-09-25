# src/test_features.py
import pandas as pd
import hopsworks

from src import config
from src.feature_store_api import get_or_create_feature_view
from src.logger import get_logger

logger = get_logger()


def load_raw_data():
    """
    Load raw historical taxi rides data from Hopsworks Feature Store.
    """
    project = hopsworks.login(
        project=config.HOPSWORKS_PROJECT_NAME,
        api_key_value=config.HOPSWORKS_API_KEY,
    )
    fs = project.get_feature_store()

    # fetch feature view (positional arg, not keyword)
    fv = get_or_create_feature_view(config.FEATURE_VIEW_METADATA)

    logger.info("📥 Fetching raw data from feature store...")
    df = fv.get_batch_data()

    logger.info(f"✅ Raw data loaded: {df.shape}")
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure pickup_ts is datetime
    if not pd.api.types.is_datetime64_any_dtype(df["pickup_ts"]):
        df["pickup_ts"] = pd.to_datetime(df["pickup_ts"], utc=True, errors="coerce")

    # Example time-based features
    df["hour_of_day"] = df["pickup_ts"].dt.hour
    df["day_of_week"] = df["pickup_ts"].dt.dayofweek
    df["month"] = df["pickup_ts"].dt.month

    # ⚡ Build lag features efficiently
    lag_features = []
    for lag in range(1, config.N_FEATURES + 1):
        lag_features.append(
            df.groupby("pickup_location_id")["rides"].shift(lag).rename(f"lag_{lag}")
        )
    lag_df = pd.concat(lag_features, axis=1)

    # Concatenate everything
    df = pd.concat([df, lag_df], axis=1)

    # Target column
    df["target_rides_next_hour"] = (
        df.groupby("pickup_location_id")["rides"].shift(-1)
    )

    # Drop rows with NaNs from shifting
    df = df.dropna().reset_index(drop=True)
    return df


def main():
    # 1. Load raw data
    raw_df = load_raw_data()

    # 2. Build features
    features_df = build_features(raw_df)

    # 3. Report shapes
    print(f"📥 Raw shape      : {raw_df.shape}")
    print(f"⚙️ Features shape : {features_df.shape}")

    # 4. Sanity check a few columns
    print("🔍 Example feature columns:", features_df.columns[:15].tolist())

    # 5. Target column check
    target_col = "target_rides_next_hour"
    print(f"🎯 Target column exists: {target_col in features_df.columns}")


if __name__ == "__main__":
    main()

# src/feature_store_api.py
import pandas as pd
import hopsworks
import hsfs
from typing import Optional, List
from src import config
from src.logger import get_logger
from src.feature_metadata import FeatureGroupConfig, FeatureViewConfig
from src.config import FEATURE_VIEW_METADATA
import os
import joblib

logger = get_logger()


def get_feature_store() -> hsfs.feature_store.FeatureStore:
    project = hopsworks.login(
        project=config.HOPSWORKS_PROJECT_NAME,
        api_key_value=config.HOPSWORKS_API_KEY,
    )
    return project.get_feature_store()


def get_feature_group(metadata: FeatureGroupConfig) -> hsfs.feature_group.FeatureGroup:
    fs = get_feature_store()
    return fs.get_feature_group(name=metadata.name, version=metadata.version)


def get_or_create_feature_group(metadata: FeatureGroupConfig) -> hsfs.feature_group.FeatureGroup:
    return get_feature_store().get_or_create_feature_group(
        name=metadata.name,
        version=metadata.version,
        description=metadata.description,
        primary_key=metadata.primary_key,
        event_time=metadata.event_time,
        online_enabled=metadata.online_enabled,
    )


def get_or_create_feature_view(metadata: FeatureViewConfig) -> hsfs.feature_view.FeatureView:
    fs = get_feature_store()
    try:
        return fs.get_feature_view(name=metadata.name, version=metadata.version)
    except Exception:
        logger.info(
            f"Feature view '{metadata.name}' v{metadata.version} not found. "
            f"Creating from feature group '{metadata.feature_group.name}'."
        )
        fg = get_or_create_feature_group(metadata.feature_group)
        query = fg.select_all()
        return fs.create_feature_view(
            name=metadata.name,
            version=metadata.version,
            query=query,
        )


def load_fallback_features() -> pd.DataFrame:
    """
    Load fallback features from local parquet file.
    """
    fallback_path = "data/transformed/ts_data_2022_01.parquet"
    if not os.path.exists(fallback_path):
        raise FileNotFoundError(f"❌ Fallback parquet not found: {fallback_path}")

    df = pd.read_parquet(fallback_path)
    logger.info(f"📂 Using fallback parquet: {fallback_path}")
    logger.info(f"📑 Columns in fallback parquet: {df.columns.tolist()}")

    # Rename pickup_hour to pickup_ts for consistency
    if "pickup_hour" in df.columns:
        df = df.rename(columns={"pickup_hour": "pickup_ts"})

    return df


def load_batch_of_features_from_store(feature_view_metadata: FeatureViewConfig, n_features: int) -> pd.DataFrame:
    fs = get_feature_store()
    fv = fs.get_feature_view(
        name=feature_view_metadata.name,
        version=feature_view_metadata.version,
    )
    df = fv.get_batch_data().sort_values("pickup_ts")

    if df.empty:
        logger.warning("⚠️ No features found in feature store")
        return pd.DataFrame()

    features_now = (
        df.groupby("pickup_location_id")
        .tail(n_features)
        .reset_index(drop=True)
    )
    return features_now


def load_predictions_from_store(from_pickup_hour=None, to_pickup_hour=None) -> pd.DataFrame:
    fs = get_feature_store()
    fg = fs.get_or_create_feature_group(
        name=config.FEATURE_GROUP_PREDICTIONS_METADATA.name,
        version=config.FEATURE_GROUP_PREDICTIONS_METADATA.version,
    )

    df = pd.DataFrame()
    try:
        print("🔎 Trying offline store read...")
        df = fg.read()
        print(f"✅ Loaded predictions from OFFLINE store, shape={df.shape}")
    except Exception as e:
        print(f"⚠️ Offline store read failed ({e}), trying online store...")
        try:
            df = fg.read(read_options={"use_hive": False})
            print(f"✅ Loaded predictions from ONLINE store, shape={df.shape}")
        except Exception as e2:
            print(f"❌ Could not read from online store either: {e2}")
            return pd.DataFrame()

    if df.empty:
        print("⚠️ Predictions DataFrame is empty!")
        return df

    print("🟢 Predictions columns:", df.columns.tolist())
    print(df.head(5))

    # Apply time filter if requested
    if from_pickup_hour and to_pickup_hour:
        print(f"⏳ Filtering predictions between {from_pickup_hour} and {to_pickup_hour}")
        before_filter = df.shape[0]
        df = df.query("pickup_hour >= @from_pickup_hour and pickup_hour <= @to_pickup_hour")
        print(f"Filtered rows: {before_filter} → {df.shape[0]}")

    return df


def log_predictions_to_store(predictions_df: pd.DataFrame):
    if predictions_df.empty:
        print("⚠️ Tried to log empty predictions DataFrame!")
        return

    print("🟢 Logging predictions, shape:", predictions_df.shape)
    print("Columns:", predictions_df.columns.tolist())
    print(predictions_df.head(5))

    fs = get_feature_store()
    fg = fs.get_or_create_feature_group(
        name=config.FEATURE_GROUP_PREDICTIONS_METADATA.name,
        version=config.FEATURE_GROUP_PREDICTIONS_METADATA.version,
    )

    # Insert into feature group (online + offline)
    fg.insert(predictions_df, write_options={"wait_for_job": False})
    print("✅ Inserted predictions into feature group.")

    if "pickup_hour" in predictions_df.columns:
        try:
            fg.materialize(
                start_time=predictions_df["pickup_hour"].min(),
                end_time=predictions_df["pickup_hour"].max(),
            )
            print("✅ Materialized predictions to OFFLINE store.")
        except Exception as e:
            print(f"⚠️ Auto-materialization failed: {e}")
    else:
        print("⚠️ Skipping materialization: no pickup_hour column found.")


# ✅ Proper main entrypoint for debugging
if __name__ == "__main__":
    print("🔗 Connecting to Hopsworks Feature Store...")
    fs = get_feature_store()
    print("✅ Connected to Feature Store")

    try:
        preds_fg = get_feature_group(config.FEATURE_GROUP_PREDICTIONS_METADATA)
        print(f"✅ Predictions FG found: {preds_fg.name} v{preds_fg.version}")
    except Exception as e:
        print(f"❌ Predictions FG not found: {e}")

    try:
        fv = get_or_create_feature_view(FEATURE_VIEW_METADATA)
        print(f"✅ Feature View available: {fv.name} v{fv.version}")
    except Exception as e:
        print(f"❌ Feature View issue: {e}")

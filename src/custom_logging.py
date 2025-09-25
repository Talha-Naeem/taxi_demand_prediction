import hopsworks
import pandas as pd
from datetime import datetime
from src import config


def log_predictions(preds_df: pd.DataFrame, current_date: datetime):
    """
    Log model predictions into the Hopsworks Feature Store.
    Ensures feature group exists before inserting.
    """
    if preds_df is None or preds_df.empty:
        print("⚠️ No predictions to log")
        return

    preds_df = preds_df.copy()
    preds_df["pickup_hour"] = pd.to_datetime(current_date).floor("H")

    project = hopsworks.login(
        project=config.HOPSWORKS_PROJECT_NAME,
        api_key_value=config.HOPSWORKS_API_KEY
    )
    fs = project.get_feature_store()

    predictions_fg = fs.get_or_create_feature_group(
        name="model_predictions_feature_group",
        version=1,
        primary_key=["pickup_location_id", "pickup_hour"],
        description="Stores model predictions for taxi demand"
    )

    predictions_fg.insert(preds_df)
    print(f"✅ Logged {len(preds_df)} predictions to feature store")

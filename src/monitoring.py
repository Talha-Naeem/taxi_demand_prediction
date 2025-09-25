# src/monitoring.py
import pandas as pd
from datetime import datetime
from src.logger import get_logger
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os

logger = get_logger()


def load_predictions(filepath="data/predictions.csv") -> pd.DataFrame:
    """Load predictions CSV."""
    df = pd.read_csv(filepath)
    logger.info(f"📥 Loaded predictions from {filepath} with shape {df.shape}")

    # Ensure timestamp is parsed
    df["pickup_ts"] = pd.to_datetime(df["pickup_ts"], unit="ms")
    return df


def evaluate_predictions(df: pd.DataFrame) -> dict:
    """Compute monitoring metrics: MAE, MSE, RMSE."""
    y_true = df["rides"]
    y_pred = df["predicted_rides_next_hour"]

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = mse ** 0.5

    metrics = {
        "timestamp": datetime.utcnow(),
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
    }
    logger.info(f"📊 Metrics: {metrics}")
    return metrics


def log_metrics(metrics: dict, filepath="data/monitoring_metrics.csv"):
    """Append metrics to CSV (keeps history)."""
    metrics_df = pd.DataFrame([metrics])

    if os.path.exists(filepath):
        old_df = pd.read_csv(filepath)
        combined = pd.concat([old_df, metrics_df], ignore_index=True)
    else:
        combined = metrics_df

    combined.to_csv(filepath, index=False)
    logger.info(f"💾 Metrics logged to {filepath}")


if __name__ == "__main__":
    predictions_df = load_predictions()
    metrics = evaluate_predictions(predictions_df)
    log_metrics(metrics)

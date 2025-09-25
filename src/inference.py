# src/inference.py
import os
import joblib
import pandas as pd
from pathlib import Path
from src.config import MODEL_NAME, MODEL_VERSION, PREDICTIONS_PATH
from src.feature_store_api import get_feature_store, get_or_create_feature_view
from src.config import FEATURE_VIEW_METADATA

from src.logger import get_logger

logger = get_logger()


def load_model():
    """Load trained model bundle from disk."""
    model_path = f"models/{MODEL_NAME}_v{MODEL_VERSION}.pkl"
    logger.info("📥 Loading model...")
    bundle = joblib.load(model_path)

    # Extract pipeline and metadata
    model = bundle["model"]
    expected_features = bundle.get("expected_features", [])
    feature_names = bundle.get("feature_names", expected_features)

    logger.info(f"✅ Model loaded with {len(feature_names)} expected features.")
    return model, feature_names



def load_features_for_inference() -> pd.DataFrame:
    """Load latest features from Hopsworks Feature Store."""
    logger.info("📊 Loading features for inference...")
    fv = get_or_create_feature_view(FEATURE_VIEW_METADATA)
    features = fv.get_batch_data()
    logger.info(f"➡️ Features shape before preprocessing: {features.shape}")
    return features


def run_inference(model, features: pd.DataFrame, expected_features: list) -> pd.DataFrame:
    """Run inference with trained model pipeline."""
    # Preprocess datetime columns (same as training)
    datetime_cols = features.select_dtypes(include=["datetime64[ns, UTC]", "datetime64[ns]"]).columns
    for col in datetime_cols:
        features[col] = features[col].astype("int64") // 10**9

    # Align columns
    features = features.reindex(columns=expected_features, fill_value=0)

    # Predict
    preds = model.predict(features)
    features["predicted_rides_next_hour"] = preds
    return features



def save_predictions(predictions_df: pd.DataFrame, path: str):
    """Save predictions to CSV (ensuring directory exists)."""
    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(save_path, index=False)
    logger.info(f"💾 Predictions saved to {save_path}")


def main():
    model, expected_features = load_model()
    features = load_features_for_inference()
    predictions_df = run_inference(model, features, expected_features)
    save_predictions(predictions_df, PREDICTIONS_PATH)
    logger.info("🚀 Inference finished successfully.")


if __name__ == "__main__":
    main()

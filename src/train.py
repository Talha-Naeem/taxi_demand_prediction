# src/train.py
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import optuna
import joblib
import os
import hopsworks
from src import config
from src.logger import get_logger
from src.feature_store_api import load_batch_of_features_from_store
from src.config import FEATURE_VIEW_METADATA, N_FEATURES, N_HYPERPARAMETER_SEARCH_TRIALS, MAX_MAE

logger = get_logger()

TARGET_COL = "target_rides_next_hour"


def fetch_features_and_target() -> pd.DataFrame:
    """Fetch features and generate target column."""
    print("📥 Fetching raw data from feature store...")
    df = load_batch_of_features_from_store(FEATURE_VIEW_METADATA, n_features=N_FEATURES)

    if df.empty:
        raise ValueError("No features loaded from feature store!")

    print(f"✅ Raw data loaded: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    # Create target column if missing
    if TARGET_COL not in df.columns:
        ride_column = "rides"
        df[TARGET_COL] = df.groupby("pickup_location_id")[ride_column].shift(-1)
        df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)

    print(f"⚙️ Features ready: {df.shape}")
    return df


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    datetime_cols = X.select_dtypes(include=["datetime64[ns, UTC]", "datetime64[ns]"]).columns
    for col in datetime_cols:
        X[col] = X[col].astype("int64") // 10**9

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    print(f"Split data: X_train={X_train.shape}, X_test={X_test.shape}")
    return X_train, X_test, y_train, y_test


def objective(trial, X_train, y_train):
    """Optuna objective for LightGBM."""
    params = {
        "objective": "regression",
        "metric": "mae",
        "boosting_type": "gbdt",
        "verbosity": -1,
        "num_leaves": trial.suggest_int("num_leaves", 31, 255),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
    }

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", lgb.LGBMRegressor(**params, n_estimators=500))
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_train)
    mae = np.mean(np.abs(y_train - y_pred))

    print(f"Trial params: {params}, MAE={mae:.4f}")

    if mae > MAX_MAE:
        raise optuna.TrialPruned()

    return mae


def save_model_with_features(model, X_train):
    model_path = f"models/{config.MODEL_NAME}_v{config.MODEL_VERSION}.pkl"
    os.makedirs("models", exist_ok=True)

    bundle = {
        "model": model,
        "expected_features": X_train.columns.tolist(),
        "feature_names": X_train.columns.tolist(),
    }

    joblib.dump(bundle, model_path)
    print(f"✅ Model and feature names saved: {model_path}")

    schema_path = os.path.join("models", "feature_schema.parquet")
    pd.DataFrame({"feature_name": X_train.columns}).to_parquet(schema_path, index=False)
    print(f"✅ Feature schema saved: {schema_path}")

    fallback_path = "data/transformed/fallback_features.parquet"
    os.makedirs("data/transformed", exist_ok=True)
    X_train.to_parquet(fallback_path, index=False)
    print(f"✅ Saved fallback features for inference: {fallback_path}")


def main():
    df = fetch_features_and_target()
    X_train, X_test, y_train, y_test = split_data(df, test_size=0.2)

    print(f"Starting hyperparameter optimization for {N_HYPERPARAMETER_SEARCH_TRIALS} trials...")
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=N_HYPERPARAMETER_SEARCH_TRIALS)

    print(f"Best parameters found: {study.best_params}")

    # Train final model
    final_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", lgb.LGBMRegressor(**study.best_params, n_estimators=500))
    ])
    final_pipeline.fit(X_train, y_train)
    print("✅ Final model trained.")

    save_model_with_features(final_pipeline, X_train)
    print("🚀 Training pipeline finished successfully.")


if __name__ == "__main__":
    main()
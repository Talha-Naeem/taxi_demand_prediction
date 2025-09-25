import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline, Pipeline
import lightgbm as lgb


def average_rides_last_4_weeks(X: pd.DataFrame) -> pd.DataFrame:
    """
    Adds one column with the average rides from:
    - 7 days ago
    - 14 days ago
    - 21 days ago
    - 28 days ago
    If any of the lag columns are missing, fills them with NaN before averaging.
    """
    X = X.copy()
    lag_hours = [7*24, 2*7*24, 3*7*24, 4*7*24]
    lag_cols = [f'rides_previous_{h}_hour' for h in lag_hours]

    # Ensure all lag columns exist
    for col in lag_cols:
        if col not in X.columns:
            X[col] = float("nan")

    X["average_rides_last_4_weeks"] = X[lag_cols].mean(axis=1)
    return X


class TemporalFeaturesEngineer(BaseEstimator, TransformerMixin):
    """
    Adds temporal features from `pickup_hour`:
    - hour of day
    - day of week
    Ensures `pickup_hour` is converted to datetime first.
    Drops the original `pickup_hour` column.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_ = X.copy()

        # Ensure pickup_hour is datetime
        if not pd.api.types.is_datetime64_any_dtype(X_["pickup_hour"]):
            X_["pickup_hour"] = pd.to_datetime(X_["pickup_hour"], errors="coerce")

        # If still NaT after conversion, fill with safe defaults
        if X_["pickup_hour"].isna().any():
            print("⚠️ Warning: Some pickup_hour values could not be parsed as datetime.")
            X_["pickup_hour"] = X_["pickup_hour"].fillna(pd.Timestamp("1970-01-01"))

        # Add temporal features
        X_["hour"] = X_["pickup_hour"].dt.hour
        X_["day_of_week"] = X_["pickup_hour"].dt.dayofweek

        return X_.drop(columns=["pickup_hour"])



def get_pipeline(**hyperparams) -> Pipeline:
    """
    Build the full preprocessing + model pipeline with fixed step names.
    """
    add_feature_average_rides_last_4_weeks = FunctionTransformer(
        average_rides_last_4_weeks, validate=False
    )
    add_temporal_features = TemporalFeaturesEngineer()

    return Pipeline([
        ("average_rides_last_4_weeks", add_feature_average_rides_last_4_weeks),
        ("temporal_features", add_temporal_features),
        ("model", lgb.LGBMRegressor(**hyperparams)),
    ])
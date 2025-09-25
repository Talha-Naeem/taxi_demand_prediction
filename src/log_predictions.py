import pandas as pd
import datetime

from src.inference import get_model_predictions, log_predictions_to_store, load_model
from src.feature_store_api import get_batch_of_features_from_store

def main():
    # 1. Load model
    model = load_model()

    # 2. Pick current UTC hour for consistency
    current_date = pd.Timestamp(
        datetime.datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    )

    # 3. Load features from feature store
    print("🔄 Loading features for:", current_date)
    features = get_batch_of_features_from_store(current_date)

    if features.empty:
        print("⚠️ No features found for this time window. Skipping prediction.")
        return

    # 4. Run predictions
    predictions_df = get_model_predictions(model, features)

    # 5. Add pickup_hour column (required for schema match in feature group)
    predictions_df["pickup_hour"] = current_date

    # 6. Log predictions into feature store
    log_predictions_to_store(predictions_df)

    print("✅ Predictions logged successfully for:", current_date)

if __name__ == "__main__":
    main()

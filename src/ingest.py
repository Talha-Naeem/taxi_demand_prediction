import pandas as pd
import hopsworks

def ingest_data_from_parquet(parquet_path: str):
    project = hopsworks.login()
    fs = project.get_feature_store()

    feature_group = fs.get_or_create_feature_group(
        name="taxi_demand_features",
        version=1,
        primary_key=["pickup_location_id", "pickup_hour"],
        event_time="pickup_ts",   # 👈 event time column
        description="Taxi demand feature group with UTC datetimes"
    )

    # ✅ Load Parquet file
    df = pd.read_parquet(parquet_path)

    # ✅ Ensure pickup_hour is datetime UTC
    df["pickup_hour"] = pd.to_datetime(df["pickup_hour"], utc=True)

    # ✅ If pickup_ts is missing, derive it from pickup_hour
    if "pickup_ts" not in df.columns:
        df["pickup_ts"] = df["pickup_hour"]

    # ✅ Final check: ensure pickup_ts is datetime UTC
    df["pickup_ts"] = pd.to_datetime(df["pickup_ts"], utc=True)

    # ✅ Insert into Hopsworks
    feature_group.insert(df)

    print(f"✅ Inserted {len(df)} rows with proper UTC datetimes into Feature Group")

if __name__ == "__main__":
    ingest_data_from_parquet(
        "/home/talha-naeem/Documents/LLM Work/taxi_demand/data/transformed/ts_data_2022_01.parquet"
    )

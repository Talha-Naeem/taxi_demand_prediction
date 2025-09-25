# src/debug_fg.py
import datetime
from hopsworks import login

# Login to your Hopsworks project
project = login()
fs = project.get_feature_store()

# Replace with your feature group name + version
FEATURE_GROUP_NAME = "time_series_hourly_feature_group"
FEATURE_GROUP_VERSION = 1

print(f"🔍 Checking Feature Group: {FEATURE_GROUP_NAME}_v{FEATURE_GROUP_VERSION}")

# Get the feature group
fg = fs.get_feature_group(FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)

# Read some rows
df = fg.read()

if df.empty:
    print("⚠️ Feature Group is EMPTY. No data ingested.")
else:
    print(f"✅ Feature Group has {len(df)} rows")

    # Show last 5 rows
    print("\n📊 Tail of Feature Group:")
    print(df.tail())

    # Check timestamp range (replace with your actual timestamp column if different)
    ts_col = "pickup_ts"
    if ts_col in df.columns:
        min_ts = df[ts_col].min()
        max_ts = df[ts_col].max()
        print(f"\n🕒 Timestamp range: {min_ts} → {max_ts}")
    else:
        print(f"⚠️ Column `{ts_col}` not found in Feature Group. Columns are: {df.columns.tolist()}")

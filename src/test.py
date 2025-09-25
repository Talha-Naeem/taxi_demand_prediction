# src/frontend_monitoring.py
import streamlit as st
import pandas as pd
import hopsworks
from datetime import datetime, timedelta, timezone

PROJECT_NAME = "taxi_demand"  # 🔹 change if needed

# --- Helper: ensure FG initialized ---
def get_predictions_fg():
    project = hopsworks.login(project=PROJECT_NAME)
    fs = project.get_feature_store()

    predictions_fg = fs.get_or_create_feature_group(
        name="model_predictions_feature_group",
        version=1,
        description="Stores model predictions for taxi demand",
        primary_key=["ride_id"],
        online_enabled=True,
        time_travel_format="HUDI"
    )

    # Try to read → if empty, insert dummy row
    try:
        df = predictions_fg.read()
        if df.empty:
            raise ValueError("Feature group is empty")
    except Exception as e:
        st.warning(f"⚠️ FG not ready ({e}). Inserting dummy row...")

        dummy = pd.DataFrame([{
            "ride_id": -1,
            "pickup_location_id": -1,
            "pickup_ts": datetime.now(timezone.utc),
            "rides": 0,
            "predicted_demand": 0.0,
        }])

        predictions_fg.insert(dummy)
        st.success("✅ Dummy row inserted. FG initialized.")

    return predictions_fg


# --- Monitoring Logic ---
def load_predictions(from_dt, to_dt):
    predictions_fg = get_predictions_fg()
    df = predictions_fg.read()

    # Convert pickup_ts if numeric (epoch ms) → datetime
    if pd.api.types.is_numeric_dtype(df["pickup_ts"]):
        df["pickup_ts"] = pd.to_datetime(df["pickup_ts"], unit="ms", utc=True)
    else:
        df["pickup_ts"] = pd.to_datetime(df["pickup_ts"], utc=True)

    # Filter by range
    df = df[(df["pickup_ts"] >= from_dt) & (df["pickup_ts"] <= to_dt)]
    return df


# --- Streamlit UI ---
def main():
    st.title("📊 Taxi Demand Monitoring")

    to_dt = datetime.now(timezone.utc)
    from_dt = to_dt - timedelta(days=2)

    st.write(f"Showing predictions from **{from_dt}** to **{to_dt}**")

    df = load_predictions(from_dt, to_dt)

    if df.empty:
        st.warning("No predictions available in this time range.")
        return

    st.dataframe(df)

    # Plot actual vs predicted demand
    agg = df.groupby("pickup_ts")[["rides", "predicted_demand"]].sum().reset_index()
    st.line_chart(agg.set_index("pickup_ts"))


if __name__ == "__main__":
    main()

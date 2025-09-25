# src/frontend.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from src.config import PREDICTIONS_PATH
from src.logger import get_logger

logger = get_logger()

# -----------------------
# Data loader
# -----------------------
@st.cache_data
def load_predictions(path: str = PREDICTIONS_PATH) -> pd.DataFrame:
    """Load predictions from CSV, normalize columns, and cleanup for UI."""
    try:
        df = pd.read_csv(path)

        # Rename predicted column for consistency
        if "predicted_rides_next_hour" in df.columns:
            df = df.rename(columns={"predicted_rides_next_hour": "predicted_demand"})

        # Convert pickup_ts → datetime
        if "pickup_ts" in df.columns:
            df["pickup_ts"] = pd.to_datetime(df["pickup_ts"], unit="ms", errors="coerce")
            df["pickup_ts"] = df["pickup_ts"].dt.floor("h")

        # Drop pickup_hour (not user-friendly)
        if "pickup_hour" in df.columns:
            df = df.drop(columns=["pickup_hour"])

        # Reorder columns for clarity
        cols = ["pickup_location_id", "pickup_ts", "rides", "predicted_demand"]
        df = df[[c for c in cols if c in df.columns]]

        return df
    except Exception as e:
        logger.error(f"❌ Failed to load predictions: {e}")
        return pd.DataFrame()

# -----------------------
# Visualization helpers
# -----------------------
def plot_overall_demand(df: pd.DataFrame):
    """Aggregate demand over time (all locations)."""
    agg = df.groupby("pickup_ts")[["rides", "predicted_demand"]].sum().reset_index()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(agg["pickup_ts"], agg["rides"], label="Actual rides", color="blue")
    ax.plot(agg["pickup_ts"], agg["predicted_demand"], label="Predicted demand", color="orange")
    ax.set_title("📊 Overall Taxi Demand (All Locations)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Number of Rides")
    ax.legend()
    st.pyplot(fig)

def plot_location_demand(df: pd.DataFrame, location_id: int):
    """Show demand trends for a specific location with scatter + trend line."""
    subset = df[df["pickup_location_id"] == location_id]

    if subset.empty:
        st.warning(f"No data for location {location_id}")
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    # Scatter for actual rides
    ax.scatter(subset["pickup_ts"], subset["rides"], label="Actual rides",
               color="blue", alpha=0.5, s=40)

    # Scatter for predicted demand
    ax.scatter(subset["pickup_ts"], subset["predicted_demand"], label="Predicted demand",
               color="orange", alpha=0.5, s=40)

    # Rolling mean (trend line) for smoother visualization
    subset_sorted = subset.sort_values("pickup_ts")
    window = max(1, len(subset_sorted) // 20)  # ~20 segments for smoothness

    ax.plot(subset_sorted["pickup_ts"],
            subset_sorted["rides"].rolling(window=window, min_periods=1).mean(),
            color="blue", linewidth=2, label="Actual trend")

    ax.plot(subset_sorted["pickup_ts"],
            subset_sorted["predicted_demand"].rolling(window=window, min_periods=1).mean(),
            color="orange", linewidth=2, label="Predicted trend")

    ax.set_title(f"📍 Taxi Demand at Location {location_id}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Number of Rides")
    ax.legend()
    st.pyplot(fig)

# -----------------------
# Main UI
# -----------------------
def main():
    st.title("🚖 Taxi Demand Prediction Dashboard")

    df = load_predictions()
    if df.empty:
        st.error("❌ No predictions available. Run inference first.")
        return

    st.subheader("📂 Raw Predictions Data")
    st.dataframe(df.head(50))  # show first 50 rows

    # Overall demand
    st.subheader("📊 Overall Demand")
    plot_overall_demand(df)

    # Per-location demand
    st.subheader("📍 Location-Specific Demand")
    location_ids = sorted(df["pickup_location_id"].unique())
    location_id = st.selectbox("Choose a pickup location", location_ids)
    plot_location_demand(df, location_id)

if __name__ == "__main__":
    main()

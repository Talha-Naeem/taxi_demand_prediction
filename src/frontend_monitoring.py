# src/frontend_monitoring.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from src.monitoring import load_predictions, evaluate_predictions



def main():
    st.set_page_config(page_title="Taxi Demand Monitoring", layout="wide")
    st.title("📈 Model Monitoring Dashboard")

    # Load data
    df = load_predictions()
    metrics = evaluate_predictions(df)

    # --- Metrics ---
    st.subheader("⚡ Model Performance Metrics (Latest Run)")
    st.write(metrics)

    # --- Historical Metrics ---
    st.subheader("📉 Monitoring Metrics Trend")
    try:
        metrics_df = pd.read_csv("data/monitoring_metrics.csv", parse_dates=["timestamp"])
        st.line_chart(metrics_df.set_index("timestamp")[["MAE", "RMSE"]])
    except FileNotFoundError:
        st.warning("No historical metrics found yet. Run monitoring.py to generate logs.")

    # --- Actual vs Predicted ---
    st.subheader("📊 Actual vs Predicted Rides Over Time")
    fig, ax = plt.subplots(figsize=(12, 6))
    df = df.sort_values("pickup_ts")
    ax.plot(df["pickup_ts"], df["rides"], label="Actual Rides", alpha=0.7)
    ax.plot(df["pickup_ts"], df["predicted_rides_next_hour"], label="Predicted Rides", alpha=0.7)
    ax.legend()
    ax.set_xlabel("Time")
    ax.set_ylabel("Number of Rides")
    st.pyplot(fig)

    # --- Error distribution ---
    st.subheader("🔍 Prediction Error Distribution")
    df["error"] = df["rides"] - df["predicted_rides_next_hour"]
    fig, ax = plt.subplots()
    ax.hist(df["error"], bins=50, alpha=0.7)
    ax.set_xlabel("Error (Actual - Predicted)")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)


if __name__ == "__main__":
    main()

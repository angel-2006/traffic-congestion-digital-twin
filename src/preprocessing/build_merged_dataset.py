import os
import pandas as pd

# Input files
TOMTOM_FILE = "data/raw/tomtom_traffic_data.csv"
SIM_SUMMARY_FILE = "data/simulation/summary.csv"

# Output file
OUTPUT_FILE = "data/processed/final_traffic_dataset.csv"


def build_dataset():
    if not os.path.exists(TOMTOM_FILE):
        print(f"TomTom file not found: {TOMTOM_FILE}")
        return

    if not os.path.exists(SIM_SUMMARY_FILE):
        print(f"Simulation summary file not found: {SIM_SUMMARY_FILE}")
        return

    # Load datasets
    traffic_df = pd.read_csv(TOMTOM_FILE)
    sim_df = pd.read_csv(SIM_SUMMARY_FILE)

    # -----------------------------
    # Clean / basic checks
    # -----------------------------
    if "timestamp" not in traffic_df.columns:
        print("Error: 'timestamp' column not found in TomTom data")
        return

    traffic_df["timestamp"] = pd.to_datetime(traffic_df["timestamp"], errors="coerce")
    traffic_df = traffic_df.dropna(subset=["timestamp"])

    # -----------------------------
    # Create TomTom traffic features
    # -----------------------------
    traffic_df["hour"] = traffic_df["timestamp"].dt.hour
    traffic_df["minute"] = traffic_df["timestamp"].dt.minute
    traffic_df["day_of_week"] = traffic_df["timestamp"].dt.dayofweek

    # Peak hour flag
    traffic_df["is_peak_hour"] = traffic_df["hour"].isin([8, 9, 10, 17, 18, 19]).astype(int)

    # Congestion ratio / jam factor proxy
    traffic_df["jamFactor"] = (
        (traffic_df["freeFlowSpeed"] - traffic_df["currentSpeed"]) / traffic_df["freeFlowSpeed"]
    ).fillna(0)

    traffic_df["jamFactor"] = traffic_df["jamFactor"].clip(lower=0, upper=1)

    # -----------------------------
    # Aggregate simulation features
    # -----------------------------
    sim_features = {
        "sim_running_avg": sim_df["running"].mean() if "running" in sim_df.columns else 0,
        "sim_waiting_avg": sim_df["waiting"].mean() if "waiting" in sim_df.columns else 0,
        "sim_meanSpeed_avg": sim_df["meanSpeed"].mean() if "meanSpeed" in sim_df.columns else 0,
        "sim_meanTravelTime_avg": sim_df["meanTravelTime"].mean() if "meanTravelTime" in sim_df.columns else 0,
    }

    for key, value in sim_features.items():
        traffic_df[key] = value

    # -----------------------------
    # Select final columns
    # -----------------------------
    final_columns = [
        "timestamp",
        "location_name",
        "currentSpeed",
        "freeFlowSpeed",
        "currentTravelTime",
        "freeFlowTravelTime",
        "confidence",
        "roadClosure",
        "hour",
        "minute",
        "day_of_week",
        "is_peak_hour",
        "jamFactor",
        "sim_running_avg",
        "sim_waiting_avg",
        "sim_meanSpeed_avg",
        "sim_meanTravelTime_avg",
    ]

    existing_columns = [col for col in final_columns if col in traffic_df.columns]
    final_df = traffic_df[existing_columns]

    os.makedirs("data/processed", exist_ok=True)
    final_df.to_csv(OUTPUT_FILE, index=False)

    print(f"Final merged dataset saved to: {OUTPUT_FILE}")
    print(f"Shape: {final_df.shape}")
    print("\nPreview:")
    print(final_df.head())


if __name__ == "__main__":
    build_dataset()
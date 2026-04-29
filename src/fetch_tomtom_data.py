import requests
import pandas as pd
import os
from datetime import datetime

# -------------------------------
# TOMTOM API SETTINGS
# -------------------------------
API_KEY = "x2xyWWG4o9tNtVizpeLzpwR2GN20Y2uW"   # <-- replace with your key
LAT = 12.9986   # Palace Grounds Junction latitude
LON = 77.5926   # Palace Grounds Junction longitude

# -------------------------------
# FILE PATHS
# -------------------------------
TOMTOM_FILE = "data/tomtom_traffic.csv"
TRAINING_FILE = "data/tomtom_training_data.csv"

os.makedirs("data", exist_ok=True)

# -------------------------------
# API URL
# -------------------------------
url = (
    f"https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"
    f"?point={LAT},{LON}&key={API_KEY}"
)

# -------------------------------
# FETCH REAL-TIME DATA
# -------------------------------
def fetch_tomtom():
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            flow = data["flowSegmentData"]

            current_speed    = flow["currentSpeed"]
            free_flow_speed  = flow["freeFlowSpeed"]
            current_tt       = flow["currentTravelTime"]
            free_flow_tt     = flow["freeFlowTravelTime"]
            road_closure     = flow["roadClosure"]

            # Derived features
            speed_ratio      = current_speed / free_flow_speed if free_flow_speed > 0 else 0
            delay            = current_tt - free_flow_tt  # seconds of extra delay

            # Congestion label derived from TomTom data
            if speed_ratio < 0.4:
                congestion_label = "High"
            elif speed_ratio < 0.7:
                congestion_label = "Medium"
            else:
                congestion_label = "Low"

            row = {
                "timestamp":          datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "currentSpeed":       current_speed,
                "freeFlowSpeed":      free_flow_speed,
                "currentTravelTime":  current_tt,
                "freeFlowTravelTime": free_flow_tt,
                "roadClosure":        road_closure,
                "speed_ratio":        round(speed_ratio, 4),
                "delay_seconds":      round(delay, 2),
                "congestion_label":   congestion_label,
            }
            return row
        else:
            print(f"[TomTom] HTTP {response.status_code}: {response.text}")
            return None
    except Exception as e:
        print(f"[TomTom] Error: {e}")
        return None


def save_row(row, filepath):
    df = pd.DataFrame([row])
    df.to_csv(filepath, mode='a', header=not os.path.exists(filepath), index=False)


if __name__ == "__main__":
    row = fetch_tomtom()
    if row:
        # 1. Save to the live display file (latest snapshot used by app.py)
        save_row(row, TOMTOM_FILE)

        # 2. Also accumulate into the training dataset
        #    The training file uses TomTom's real-world speed features as X
        #    and congestion_label as y, so the model learns from real data.
        training_row = {
            "currentSpeed":       row["currentSpeed"],
            "freeFlowSpeed":      row["freeFlowSpeed"],
            "speed_ratio":        row["speed_ratio"],
            "delay_seconds":      row["delay_seconds"],
            "roadClosure":        int(row["roadClosure"]),
            "congestion_label":   row["congestion_label"],
        }
        save_row(training_row, TRAINING_FILE)

        print("✅ TomTom live traffic fetched and saved.")
        print(pd.DataFrame([row]).to_string(index=False))
    else:
        print("❌ Failed to fetch TomTom data.")
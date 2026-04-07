import requests
import pandas as pd
import os
from datetime import datetime

# -------------------------------
# TOMTOM API SETTINGS
# -------------------------------
API_KEY = "x2xyWWG4o9tNtVizpeLzpwR2GN20Y2uW"   # <-- replace this
LAT = 12.9986   # Palace Grounds Junction latitude
LON = 77.5926   # Palace Grounds Junction longitude

# -------------------------------
# FILE PATH
# -------------------------------
output_file = "data/tomtom_traffic.csv"

# -------------------------------
# API URL
# -------------------------------
url = f"https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json?point={LAT},{LON}&key={API_KEY}"

# -------------------------------
# FETCH DATA
# -------------------------------
response = requests.get(url)

if response.status_code == 200:
    data = response.json()

    flow_data = data["flowSegmentData"]

    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "currentSpeed": flow_data["currentSpeed"],
        "freeFlowSpeed": flow_data["freeFlowSpeed"],
        "currentTravelTime": flow_data["currentTravelTime"],
        "freeFlowTravelTime": flow_data["freeFlowTravelTime"],
        "roadClosure": flow_data["roadClosure"]
    }

    new_df = pd.DataFrame([row])

    os.makedirs("data", exist_ok=True)

    if os.path.exists(output_file):
        old_df = pd.read_csv(output_file)
        combined_df = pd.concat([old_df, new_df], ignore_index=True)
        combined_df.to_csv(output_file, index=False)
    else:
        new_df.to_csv(output_file, index=False)

    print("\nTomTom live traffic fetched successfully!")
    print(new_df)

else:
    print("Error fetching TomTom data")
    print("Status code:", response.status_code)
    print("Response:", response.text)
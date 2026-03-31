import requests
import pandas as pd
from datetime import datetime

from src.data_collection.config import TOMTOM_API_KEY
from src.data_collection.locations import TRAFFIC_POINTS

BASE_URL = "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"


def fetch_traffic_data(lat, lon):
    params = {
        "key": TOMTOM_API_KEY,
        "point": f"{lat},{lon}",
        "unit": "kmph",
        "thickness": 10
    }

    response = requests.get(BASE_URL, params=params)

    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error fetching data for {lat}, {lon}: {response.status_code}")
        print(response.text)
        return None


def extract_relevant_fields(raw_data, location_name, lat, lon):
    if not raw_data or "flowSegmentData" not in raw_data:
        return None

    segment = raw_data["flowSegmentData"]

    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "location_name": location_name,
        "latitude": lat,
        "longitude": lon,
        "frc": segment.get("frc"),
        "currentSpeed": segment.get("currentSpeed"),
        "freeFlowSpeed": segment.get("freeFlowSpeed"),
        "currentTravelTime": segment.get("currentTravelTime"),
        "freeFlowTravelTime": segment.get("freeFlowTravelTime"),
        "confidence": segment.get("confidence"),
        "roadClosure": segment.get("roadClosure")
    }


def collect_all_locations():
    collected_data = []

    for point in TRAFFIC_POINTS:
        print(f"Fetching data for {point['location_name']}...")

        raw_data = fetch_traffic_data(point["lat"], point["lon"])
        extracted = extract_relevant_fields(
            raw_data,
            point["location_name"],
            point["lat"],
            point["lon"]
        )

        if extracted:
            collected_data.append(extracted)

    return collected_data


def save_to_csv(data):
    if not data:
        print("No data to save.")
        return

    df = pd.DataFrame(data)
    output_path = "data/raw/tomtom_traffic_data.csv"
    df.to_csv(output_path, index=False)

    print(f"\nData saved successfully to: {output_path}")
    print(df)


if __name__ == "__main__":
    all_data = collect_all_locations()
    save_to_csv(all_data)
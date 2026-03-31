import requests
import json
from src.data_collection.config import TOMTOM_API_KEY

# Example test point (Bangalore area)
LAT = 12.9716
LON = 77.5946

BASE_URL = "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"

params = {
    "key": TOMTOM_API_KEY,
    "point": f"{LAT},{LON}",
    "unit": "kmph",
    "thickness": 10
}

response = requests.get(BASE_URL, params=params)

print("Status Code:", response.status_code)

if response.status_code == 200:
    data = response.json()
    print("API call successful!\n")
    print(json.dumps(data, indent=4))
else:
    print("Error:", response.text)
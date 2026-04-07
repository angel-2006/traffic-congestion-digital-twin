import pandas as pd
import joblib

# -------------------------------
# FILE PATHS
# -------------------------------
data_file = "data/traffic_data.csv"
model_file = "models/future_congestion_model.pkl"

# -------------------------------
# LOAD DATA
# -------------------------------
df = pd.read_csv(data_file)

# -------------------------------
# LOAD TRAINED MODEL
# -------------------------------
model = joblib.load(model_file)

# -------------------------------
# TAKE LATEST TRAFFIC STATE
# -------------------------------
latest_row = df.iloc[-1]

input_data = pd.DataFrame([{
    "vehicle_count": latest_row["vehicle_count"],
    "avg_speed": latest_row["avg_speed"],
    "total_waiting_time": latest_row["total_waiting_time"],
    "stopped_vehicles": latest_row["stopped_vehicles"]
}])

# -------------------------------
# PREDICT FUTURE CONGESTION
# -------------------------------
prediction = model.predict(input_data)[0]

# -------------------------------
# PRINT RESULTS
# -------------------------------
print("\n========== FUTURE CONGESTION PREDICTION ==========\n")

print("Current Traffic State Used for Prediction")
print("------------------------------------------")
print(f"Vehicle Count       : {latest_row['vehicle_count']}")
print(f"Average Speed       : {latest_row['avg_speed']:.2f} km/h")
print(f"Total Waiting Time  : {latest_row['total_waiting_time']:.2f}")
print(f"Stopped Vehicles    : {latest_row['stopped_vehicles']}")

print("\nPredicted Future Congestion")
print("------------------------------------------")
print(f"Congestion after next few minutes: {prediction}")

print("\n=================================================\n")
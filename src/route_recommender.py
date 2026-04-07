import pandas as pd
import joblib

# -------------------------------
# FILE PATHS
# -------------------------------
data_file = "data/traffic_data.csv"
model_file = "models/future_congestion_model.pkl"

# -------------------------------
# LOAD DATA AND MODEL
# -------------------------------
df = pd.read_csv(data_file)
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
# SUGGESTIONS BASED ON PREDICTION
# -------------------------------
if prediction == "Low":
    suggestions = [
        "Traffic is smooth at Palace Grounds Junction.",
        "No rerouting required.",
        "Continue normal traffic monitoring."
    ]

elif prediction == "Medium":
    suggestions = [
        "Moderate congestion expected at Palace Grounds Junction.",
        "Monitor incoming traffic closely.",
        "Prepare to divert overflow vehicles through nearby internal roads.",
        "Issue mild congestion advisory if needed."
    ]

elif prediction == "High":
    suggestions = [
        "High congestion expected at Palace Grounds Junction.",
        "Avoid directing additional traffic into the main junction approach.",
        "Divert vehicles through alternate internal roads near Palace Grounds.",
        "Reduce inflow to the primary corridor if possible.",
        "Issue congestion alert for traffic management."
    ]

else:
    suggestions = [
        "No suggestion available."
    ]

# -------------------------------
# PRINT RESULTS
# -------------------------------
print("\n========== ROUTE / TRAFFIC RECOMMENDATION ==========\n")

print("Current Traffic State")
print("------------------------------------------")
print(f"Vehicle Count       : {latest_row['vehicle_count']}")
print(f"Average Speed       : {latest_row['avg_speed']:.2f} km/h")
print(f"Total Waiting Time  : {latest_row['total_waiting_time']:.2f}")
print(f"Stopped Vehicles    : {latest_row['stopped_vehicles']}")

print("\nPredicted Future Congestion")
print("------------------------------------------")
print(f"Congestion after next few minutes: {prediction}")

print("\nRecommended Actions")
print("------------------------------------------")
for i, suggestion in enumerate(suggestions, start=1):
    print(f"{i}. {suggestion}")

print("\n====================================================\n")
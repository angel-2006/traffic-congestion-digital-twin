import pandas as pd
import joblib
import subprocess

# -------------------------------
# FILE PATHS
# -------------------------------
sumo_file = "data/traffic_data.csv"
tomtom_file = "data/tomtom_traffic.csv"
model_file = "models/future_congestion_model.pkl"

# -------------------------------
# STEP 1: FETCH FRESH TOMTOM DATA
# -------------------------------
print("\nFetching fresh live traffic from TomTom...\n")
subprocess.run(["python", "src/fetch_tomtom_data.py"])

# -------------------------------
# LOAD DATA AND MODEL
# -------------------------------
sumo_df = pd.read_csv(sumo_file)
tomtom_df = pd.read_csv(tomtom_file)
model = joblib.load(model_file)

# -------------------------------
# GET LATEST REAL TRAFFIC (TOMTOM)
# -------------------------------
latest_tomtom = tomtom_df.iloc[-1]

current_speed = latest_tomtom["currentSpeed"]
free_flow_speed = latest_tomtom["freeFlowSpeed"]
travel_time = latest_tomtom["currentTravelTime"]

# -------------------------------
# CLASSIFY CURRENT REAL TRAFFIC STATE
# -------------------------------
speed_ratio = current_speed / free_flow_speed if free_flow_speed != 0 else 0

if speed_ratio < 0.4:
    live_traffic_state = "High"
elif speed_ratio < 0.7:
    live_traffic_state = "Medium"
else:
    live_traffic_state = "Low"

# -------------------------------
# GET LATEST DIGITAL TWIN STATE (SUMO)
# -------------------------------
latest_sumo = sumo_df.iloc[-1]

vehicle_count = latest_sumo["vehicle_count"]
avg_speed = latest_sumo["avg_speed"]
total_waiting_time = latest_sumo["total_waiting_time"]
stopped_vehicles = latest_sumo["stopped_vehicles"]

# -------------------------------
# FUTURE CONGESTION PREDICTION
# -------------------------------
input_data = pd.DataFrame([{
    "vehicle_count": vehicle_count,
    "avg_speed": avg_speed,
    "total_waiting_time": total_waiting_time,
    "stopped_vehicles": stopped_vehicles
}])

predicted_congestion = model.predict(input_data)[0]

# -------------------------------
# ROUTE / MITIGATION SUGGESTIONS
# -------------------------------
if predicted_congestion == "Low":
    suggestions = [
        "Traffic is smooth at Palace Grounds Junction.",
        "No rerouting required.",
        "Continue normal monitoring."
    ]

elif predicted_congestion == "Medium":
    suggestions = [
        "Moderate congestion expected at Palace Grounds Junction.",
        "Monitor incoming traffic closely.",
        "Prepare to divert overflow vehicles through nearby internal roads.",
        "Issue mild congestion advisory if needed."
    ]

elif predicted_congestion == "High":
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
# WHAT-IF ANALYSIS EXAMPLE
# -------------------------------
what_if_input = pd.DataFrame([{
    "vehicle_count": vehicle_count + 20,
    "avg_speed": max(avg_speed - 5, 0),
    "total_waiting_time": total_waiting_time + 50,
    "stopped_vehicles": stopped_vehicles + 5
}])

what_if_prediction = model.predict(what_if_input)[0]

# -------------------------------
# PRINT FINAL PROTOTYPE OUTPUT
# -------------------------------
print("\n========== PALACE GROUNDS TRAFFIC DIGITAL TWIN ==========\n")

print("1) REAL TRAFFIC INPUT (TomTom)")
print("------------------------------------------------------")
if "timestamp" in latest_tomtom:
    print(f"Timestamp           : {latest_tomtom['timestamp']}")
print(f"Current Speed       : {current_speed} km/h")
print(f"Free Flow Speed     : {free_flow_speed} km/h")
print(f"Current Travel Time : {travel_time} sec")
print(f"Current Real Traffic State: {live_traffic_state}")

print("\n2) DIGITAL TWIN STATE (SUMO + TraCI)")
print("------------------------------------------------------")
print(f"Vehicle Count       : {vehicle_count}")
print(f"Average Speed       : {avg_speed:.2f} km/h")
print(f"Total Waiting Time  : {total_waiting_time:.2f}")
print(f"Stopped Vehicles    : {stopped_vehicles}")

print("\n3) FUTURE CONGESTION PREDICTION")
print("------------------------------------------------------")
print(f"Predicted Congestion after next few minutes: {predicted_congestion}")

print("\n4) ROUTE / TRAFFIC MANAGEMENT SUGGESTIONS")
print("------------------------------------------------------")
for i, suggestion in enumerate(suggestions, start=1):
    print(f"{i}. {suggestion}")

print("\n5) WHAT-IF ANALYSIS EXAMPLE")
print("------------------------------------------------------")
print("If traffic increases artificially:")
print(f"Vehicle Count       : {vehicle_count + 20}")
print(f"Average Speed       : {max(avg_speed - 5, 0):.2f} km/h")
print(f"Total Waiting Time  : {total_waiting_time + 50:.2f}")
print(f"Stopped Vehicles    : {stopped_vehicles + 5}")
print(f"Predicted Future Congestion: {what_if_prediction}")

print("\n=========================================================\n")
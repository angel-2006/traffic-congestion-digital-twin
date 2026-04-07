import pandas as pd

# -------------------------------
# LOAD SUMO DATA
# -------------------------------
sumo_file = "data/traffic_data.csv"
tomtom_file = "data/tomtom_traffic.csv"

sumo_df = pd.read_csv(sumo_file)
tomtom_df = pd.read_csv(tomtom_file)

# -------------------------------
# USE LATEST DATA
# -------------------------------
latest_sumo = sumo_df.iloc[-1]
latest_tomtom = tomtom_df.iloc[-1]

# -------------------------------
# EXTRACT VALUES
# -------------------------------
sumo_avg_speed = latest_sumo["avg_speed"]
sumo_vehicle_count = latest_sumo["vehicle_count"]
sumo_waiting_time = latest_sumo["total_waiting_time"]

tomtom_current_speed = latest_tomtom["currentSpeed"]
tomtom_free_speed = latest_tomtom["freeFlowSpeed"]
tomtom_travel_time = latest_tomtom["currentTravelTime"]

# -------------------------------
# SUMO CONGESTION LOGIC
# -------------------------------
if sumo_avg_speed < 5:
    sumo_congestion = "High"
elif sumo_avg_speed < 15:
    sumo_congestion = "Medium"
else:
    sumo_congestion = "Low"

# -------------------------------
# TOMTOM CONGESTION LOGIC
# -------------------------------
speed_ratio = tomtom_current_speed / tomtom_free_speed if tomtom_free_speed != 0 else 0

if speed_ratio < 0.4:
    tomtom_congestion = "High"
elif speed_ratio < 0.7:
    tomtom_congestion = "Medium"
else:
    tomtom_congestion = "Low"

# -------------------------------
# VALIDATION RESULT
# -------------------------------
if sumo_congestion == tomtom_congestion:
    validation_result = "MATCH"
else:
    validation_result = "MISMATCH"

# -------------------------------
# PRINT RESULTS
# -------------------------------
print("\n========== DIGITAL TWIN VALIDATION ==========\n")

print("SUMO (Simulated Traffic)")
print("------------------------")
print(f"Vehicle Count       : {sumo_vehicle_count}")
print(f"Average Speed       : {sumo_avg_speed:.2f} km/h")
print(f"Total Waiting Time  : {sumo_waiting_time:.2f}")
print(f"Predicted Congestion: {sumo_congestion}")

print("\nTomTom (Real Traffic)")
print("------------------------")
print(f"Current Speed       : {tomtom_current_speed} km/h")
print(f"Free Flow Speed     : {tomtom_free_speed} km/h")
print(f"Current Travel Time : {tomtom_travel_time}")
print(f"Real Congestion     : {tomtom_congestion}")

print("\nValidation Result")
print("------------------------")
print(f"Digital Twin Status : {validation_result}")

if validation_result == "MATCH":
    print("The SUMO digital twin is closely reflecting real-world traffic conditions.")
else:
    print("The SUMO digital twin differs from current real-world traffic conditions.")

print("\n=============================================\n")
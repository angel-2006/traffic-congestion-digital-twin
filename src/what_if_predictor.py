import pandas as pd
import joblib

# -------------------------------
# LOAD MODEL
# -------------------------------
model_file = "models/future_congestion_model.pkl"
model = joblib.load(model_file)

# -------------------------------
# USER INPUTS
# -------------------------------
print("\n========== WHAT-IF TRAFFIC ANALYSIS ==========\n")

vehicle_count = int(input("Enter vehicle count: "))
avg_speed = float(input("Enter average speed (km/h): "))
total_waiting_time = float(input("Enter total waiting time: "))
stopped_vehicles = int(input("Enter stopped vehicles: "))

# -------------------------------
# CREATE INPUT DATAFRAME
# -------------------------------
input_data = pd.DataFrame([{
    "vehicle_count": vehicle_count,
    "avg_speed": avg_speed,
    "total_waiting_time": total_waiting_time,
    "stopped_vehicles": stopped_vehicles
}])

# -------------------------------
# PREDICT FUTURE CONGESTION
# -------------------------------
prediction = model.predict(input_data)[0]

# -------------------------------
# PRINT RESULT
# -------------------------------
print("\n------ INPUT TRAFFIC STATE ------")
print(f"Vehicle Count       : {vehicle_count}")
print(f"Average Speed       : {avg_speed:.2f} km/h")
print(f"Total Waiting Time  : {total_waiting_time:.2f}")
print(f"Stopped Vehicles    : {stopped_vehicles}")

print("\n------ PREDICTION RESULT ------")
print(f"Predicted Congestion after next few minutes: {prediction}")

print("\n=============================================\n")
import joblib
import pandas as pd

# Load trained model
model = joblib.load("models/congestion_model.pkl")

# Example new traffic input
sample_data = pd.DataFrame([{
    "vehicle_count": 55,
    "avg_speed": 3.8,
    "total_waiting_time": 120.5,
    "stopped_vehicles": 18
}])

prediction = model.predict(sample_data)

print("Predicted Congestion Level:", prediction[0])
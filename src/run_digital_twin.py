import traci
import joblib
import pandas as pd

# -------------------------------
# LOAD MODEL
# -------------------------------
model = joblib.load("models/future_congestion_model.pkl")

# -------------------------------
# COLOR FUNCTION
# -------------------------------
def get_color(congestion):
    if congestion == "Low":
        return (0, 255, 0)      # Green
    elif congestion == "Medium":
        return (255, 255, 0)    # Yellow
    else:
        return (255, 0, 0)      # Red

# -------------------------------
# LOAD LATEST SUMO DATA
# -------------------------------
def get_latest_sumo_data():
    df = pd.read_csv("data/traffic_data.csv")
    return df.iloc[-1]

# -------------------------------
# START SUMO WITH TRACI
# -------------------------------
sumoCmd = ["sumo-gui", "sumo/config.sumocfg"]
traci.start(sumoCmd)

edges = traci.edge.getIDList()

step = 0

while step < 300:
    traci.simulationStep()

    # Every 10 steps update prediction
    if step % 10 == 0:
        data = get_latest_sumo_data()

        input_data = [[
            data["vehicle_count"],
            data["avg_speed"],
            data["total_waiting_time"],
            data["stopped_vehicles"]
        ]]

        prediction = model.predict(input_data)[0]

        color = get_color(prediction)

        # Apply color to all edges
        for edge in edges:
            traci.edge.setColor(edge, color)

    step += 1

traci.close()
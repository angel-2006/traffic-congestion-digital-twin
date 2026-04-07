import os
import sys
import csv
import traci

# -------------------------------
# SUMO setup
# -------------------------------
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    raise EnvironmentError("SUMO_HOME is not set properly")

SUMO_CONFIG = "sumo/config.sumocfg"
SUMO_BINARY = "sumo-gui"   # change to "sumo" if you don't want GUI

sumo_cmd = [SUMO_BINARY, "-c", SUMO_CONFIG]

# -------------------------------
# Start SUMO
# -------------------------------
traci.start(sumo_cmd)

# -------------------------------
# Output CSV
# -------------------------------
output_file = "data/traffic_data.csv"

with open(output_file, mode="w", newline="") as f:
    writer = csv.writer(f)

    # CSV header
    writer.writerow([
        "step",
        "vehicle_count",
        "avg_speed",
        "total_waiting_time",
        "stopped_vehicles",
        "congestion_label"
    ])

    step = 0
    max_steps = 1000   # same as your simulation end time

    while step < max_steps:
        traci.simulationStep()

        vehicle_ids = traci.vehicle.getIDList()
        vehicle_count = len(vehicle_ids)

        total_speed = 0
        total_waiting_time = 0
        stopped_vehicles = 0

        for vid in vehicle_ids:
            speed = traci.vehicle.getSpeed(vid)
            waiting_time = traci.vehicle.getWaitingTime(vid)

            total_speed += speed
            total_waiting_time += waiting_time

            if speed < 0.1:
                stopped_vehicles += 1

        avg_speed = total_speed / vehicle_count if vehicle_count > 0 else 0

        # -------------------------------
        # Simple congestion labeling logic
        # -------------------------------
        if vehicle_count > 80 and avg_speed < 5:
            congestion_label = "High"
        elif vehicle_count > 40 and avg_speed < 10:
            congestion_label = "Medium"
        else:
            congestion_label = "Low"

        # Save row
        writer.writerow([
            step,
            vehicle_count,
            round(avg_speed, 2),
            round(total_waiting_time, 2),
            stopped_vehicles,
            congestion_label
        ])

        print(
            f"Step {step} | Vehicles: {vehicle_count} | "
            f"Avg Speed: {avg_speed:.2f} | Waiting: {total_waiting_time:.2f} | "
            f"Stopped: {stopped_vehicles} | Label: {congestion_label}"
        )

        step += 1

traci.close()
print(f"\nTraffic data saved successfully to: {output_file}")
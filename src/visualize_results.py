import os
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------
# CREATE OUTPUT FOLDER
# -------------------------------
os.makedirs("outputs", exist_ok=True)

# -------------------------------
# LOAD DATA
# -------------------------------
sumo_df = pd.read_csv("data/traffic_data.csv")
tomtom_df = pd.read_csv("data/tomtom_traffic.csv")

# -------------------------------
# GRAPH 1: Vehicle Count Over Time
# -------------------------------
plt.figure(figsize=(10, 5))
plt.plot(sumo_df["step"], sumo_df["vehicle_count"], marker='o')
plt.title("SUMO Vehicle Count Over Time")
plt.xlabel("Simulation Step")
plt.ylabel("Vehicle Count")
plt.grid(True)
plt.tight_layout()
plt.savefig("outputs/vehicle_count_over_time.png")
plt.show()

# -------------------------------
# GRAPH 2: Average Speed Over Time
# -------------------------------
plt.figure(figsize=(10, 5))
plt.plot(sumo_df["step"], sumo_df["avg_speed"], marker='o')
plt.title("SUMO Average Speed Over Time")
plt.xlabel("Simulation Step")
plt.ylabel("Average Speed (km/h)")
plt.grid(True)
plt.tight_layout()
plt.savefig("outputs/avg_speed_over_time.png")
plt.show()

# -------------------------------
# GRAPH 3: Waiting Time Over Time
# -------------------------------
plt.figure(figsize=(10, 5))
plt.plot(sumo_df["step"], sumo_df["total_waiting_time"], marker='o')
plt.title("SUMO Total Waiting Time Over Time")
plt.xlabel("Simulation Step")
plt.ylabel("Total Waiting Time")
plt.grid(True)
plt.tight_layout()
plt.savefig("outputs/waiting_time_over_time.png")
plt.show()

# -------------------------------
# GRAPH 4: TomTom Current Speed Over Time
# -------------------------------
plt.figure(figsize=(10, 5))
plt.plot(range(len(tomtom_df)), tomtom_df["currentSpeed"], marker='o')
plt.title("TomTom Current Speed Over Time")
plt.xlabel("TomTom Observation Number")
plt.ylabel("Current Speed (km/h)")
plt.grid(True)
plt.tight_layout()
plt.savefig("outputs/tomtom_speed_over_time.png")
plt.show()

# -------------------------------
# GRAPH 5: SUMO vs TomTom Speed Comparison
# -------------------------------
sumo_last_n = sumo_df["avg_speed"].tail(len(tomtom_df)).reset_index(drop=True)
tomtom_speed = tomtom_df["currentSpeed"].reset_index(drop=True)

plt.figure(figsize=(10, 5))
plt.plot(range(len(sumo_last_n)), sumo_last_n, marker='o', label="SUMO Avg Speed")
plt.plot(range(len(tomtom_speed)), tomtom_speed, marker='s', label="TomTom Current Speed")
plt.title("SUMO vs TomTom Speed Comparison")
plt.xlabel("Observation Index")
plt.ylabel("Speed (km/h)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("outputs/sumo_vs_tomtom_speed.png")
plt.show()

print("\nAll graphs generated successfully!")
print("Saved inside: outputs/")
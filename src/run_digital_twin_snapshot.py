import os
import time
import traci

# -------------------------------
# FILE PATHS
# -------------------------------
SUMO_CONFIG = "sumo/config.sumocfg"
OUTPUT_IMAGE = "simulation_frames/latest_simulation.png"

# -------------------------------
# CREATE OUTPUT FOLDER
# -------------------------------
os.makedirs("simulation_frames", exist_ok=True)

# -------------------------------
# START SUMO GUI WITH AUTO-RUN
# -------------------------------
sumoBinary = "sumo-gui"
sumoCmd = [
    sumoBinary,
    "-c", SUMO_CONFIG,
    "--start"
]

print("Starting SUMO Digital Twin...")
traci.start(sumoCmd)

# -------------------------------
# LET GUI LOAD PROPERLY
# -------------------------------
time.sleep(2)

# -------------------------------
# RUN SIMULATION AUTOMATICALLY
# -------------------------------
for step in range(200):
    traci.simulationStep()
    time.sleep(0.03)   # small delay so GUI visibly renders

print("Simulation stepped successfully.")

# -------------------------------
# WAIT BEFORE SCREENSHOT
# -------------------------------
time.sleep(2)

# -------------------------------
# TAKE SNAPSHOT
# -------------------------------
try:
    traci.gui.screenshot("View #0", OUTPUT_IMAGE)
    print(f"Snapshot saved successfully: {OUTPUT_IMAGE}")
except Exception as e:
    print("Screenshot failed:", e)

# -------------------------------
# KEEP WINDOW OPEN FOR CHECKING (OPTIONAL)
# -------------------------------
time.sleep(2)

# -------------------------------
# CLOSE SUMO
# -------------------------------
traci.close()
print("SUMO closed.")
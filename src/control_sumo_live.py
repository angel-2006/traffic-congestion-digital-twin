"""
control_sumo_live.py
--------------------
Launches SUMO-GUI and colors road edges in real time using data fetched from
the TomTom API — replicating the Google Maps red / orange / green style.

Color scheme:
    🔴 Red    — speed_ratio < 0.4  (heavy congestion)
    🟠 Orange — speed_ratio < 0.7  (moderate traffic)
    🟢 Green  — speed_ratio >= 0.7 (free flow)

The TomTom speed ratio is applied globally to all edges in the network because
the TomTom endpoint returns a single flow value for the monitored road segment
(Palace Grounds Junction).  If you have multiple TomTom segments, extend
`color_edges_by_tomtom()` to map each segment to the corresponding SUMO edges.

Usage:
    python src/control_sumo_live.py
"""

import os
import sys
import time
import requests
import traci
from datetime import datetime

# -----------------------------------------------------------------------
# TOMTOM SETTINGS  (same key / location as fetch_tomtom_data.py)
# -----------------------------------------------------------------------
TOMTOM_API_KEY = "x2xyWWG4o9tNtVizpeLzpwR2GN20Y2uW"   # <-- your key
LAT = 12.9986
LON = 77.5926
TOMTOM_URL = (
    f"https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"
    f"?point={LAT},{LON}&key={TOMTOM_API_KEY}"
)

# -----------------------------------------------------------------------
# SUMO SETTINGS
# -----------------------------------------------------------------------
SUMO_CONFIG = "sumo/config.sumocfg"
SUMO_CMD    = ["sumo-gui", "-c", SUMO_CONFIG, "--start", "--quit-on-end"]

# How many simulation steps to run
MAX_STEPS = 1000

# Re-fetch TomTom data every N simulation steps
TOMTOM_REFRESH_STEPS = 50

# -----------------------------------------------------------------------
# COLOR HELPERS
# -----------------------------------------------------------------------
# RGBA tuples accepted by traci.edge.setColor
COLOR_RED    = (220, 50,  50,  255)   # Heavy congestion
COLOR_ORANGE = (255, 165,  0,  255)   # Moderate traffic
COLOR_GREEN  = (50,  200,  50, 255)   # Free flow
COLOR_GREY   = (150, 150, 150, 255)   # Unknown / no data

def speed_ratio_to_color(speed_ratio: float) -> tuple:
    if speed_ratio < 0.4:
        return COLOR_RED
    elif speed_ratio < 0.7:
        return COLOR_ORANGE
    else:
        return COLOR_GREEN

# -----------------------------------------------------------------------
# TOMTOM FETCH (non-blocking; returns None on failure)
# -----------------------------------------------------------------------
def fetch_tomtom_speed_ratio() -> float | None:
    try:
        resp = requests.get(TOMTOM_URL, timeout=8)
        if resp.status_code == 200:
            flow = resp.json()["flowSegmentData"]
            current   = flow["currentSpeed"]
            free_flow = flow["freeFlowSpeed"]
            ratio = current / free_flow if free_flow > 0 else 1.0
            print(
                f"[TomTom {datetime.now().strftime('%H:%M:%S')}] "
                f"speed={current} km/h  free_flow={free_flow} km/h  ratio={ratio:.2f}"
            )
            return ratio
        else:
            print(f"[TomTom] HTTP {resp.status_code}")
    except Exception as e:
        print(f"[TomTom] Fetch error: {e}")
    return None

# -----------------------------------------------------------------------
# EDGE COLORING
# -----------------------------------------------------------------------
def color_edges_by_tomtom(edges: list, speed_ratio: float | None):
    """Apply a single TomTom speed ratio colour to every edge in the network."""
    color = speed_ratio_to_color(speed_ratio) if speed_ratio is not None else COLOR_GREY
    for edge in edges:
        traci.edge.setColor(edge, color)

def color_edges_by_sumo(edges: list):
    """
    Fallback: colour each edge individually using its own SUMO mean speed.
    Uses the same thresholds as the TomTom ratio but based on raw speed values
    (km/h equivalent after converting from m/s).
    """
    for edge in edges:
        speed_ms  = traci.edge.getLastStepMeanSpeed(edge)   # m/s
        speed_kmh = speed_ms * 3.6

        if speed_kmh < 15:
            color = COLOR_RED
        elif speed_kmh < 35:
            color = COLOR_ORANGE
        else:
            color = COLOR_GREEN

        traci.edge.setColor(edge, color)

# -----------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------
def run():
    print("🚀 Starting SUMO-GUI digital twin …")
    traci.start(SUMO_CMD)

    edges = traci.edge.getIDList()
    print(f"   Network has {len(edges)} edges.")

    # Initial TomTom fetch
    speed_ratio = fetch_tomtom_speed_ratio()

    # Set initial GUI viewport
    try:
        view = traci.gui.getIDList()[0]
        traci.gui.setZoom(view, 2500)
        traci.gui.setOffset(view, 0, 0)
    except Exception:
        pass  # GUI might not expose a view immediately; non-fatal

    step = 0
    while step < MAX_STEPS:
        traci.simulationStep()

        # Refresh TomTom data periodically
        if step % TOMTOM_REFRESH_STEPS == 0:
            new_ratio = fetch_tomtom_speed_ratio()
            if new_ratio is not None:
                speed_ratio = new_ratio

        # Apply colour: prefer TomTom ratio; fall back to SUMO per-edge speed
        if speed_ratio is not None:
            color_edges_by_tomtom(edges, speed_ratio)
        else:
            color_edges_by_sumo(edges)

        step += 1

    traci.close()
    print("✅ Simulation complete.")


if __name__ == "__main__":
    run()
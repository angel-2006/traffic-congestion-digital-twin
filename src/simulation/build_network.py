import os
import subprocess

OSM_INPUT = "sumo/maps/silk_board_corridor.osm"
NET_OUTPUT = "sumo/maps/silk_board_corridor.net.xml"


def build_sumo_network():
    if not os.path.exists(OSM_INPUT):
        print(f"OSM file not found: {OSM_INPUT}")
        return

    os.makedirs("sumo/maps", exist_ok=True)

    command = [
        "netconvert",
        "--osm-files", OSM_INPUT,
        "--output-file", NET_OUTPUT,
        "--geometry.remove",
        "--roundabouts.guess",
        "--ramps.guess",
        "--junctions.join",
        "--tls.guess-signals",
        "--tls.discard-simple",
        "--lefthand",
        "--no-turnarounds"
    ]

    print("Building SUMO road network...")
    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"Network created successfully: {NET_OUTPUT}")
    else:
        print("Error while building network:")
        print(result.stderr)


if __name__ == "__main__":
    build_sumo_network()
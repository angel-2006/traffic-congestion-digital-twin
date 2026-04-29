import os
import xml.etree.ElementTree as ET
import pandas as pd

SUMMARY_XML = "sumo/outputs/summary.xml"
TRIPINFO_XML = "sumo/outputs/tripinfo.xml"

SUMMARY_CSV = "data/simulation/summary.csv"
TRIPINFO_CSV = "data/simulation/tripinfo.csv"


def parse_summary():
    if not os.path.exists(SUMMARY_XML):
        print(f"Summary file not found: {SUMMARY_XML}")
        return

    tree = ET.parse(SUMMARY_XML)
    root = tree.getroot()

    rows = []

    for step in root.findall("step"):
        rows.append({
            "time": float(step.attrib.get("time", 0)),
            "loaded": int(step.attrib.get("loaded", 0)),
            "inserted": int(step.attrib.get("inserted", 0)),
            "running": int(step.attrib.get("running", 0)),
            "waiting": int(step.attrib.get("waiting", 0)),
            "ended": int(step.attrib.get("ended", 0)),
            "meanWaitingTime": float(step.attrib.get("meanWaitingTime", 0)),
            "meanTravelTime": float(step.attrib.get("meanTravelTime", 0)),
            "meanSpeed": float(step.attrib.get("meanSpeed", 0))
        })

    df = pd.DataFrame(rows)
    os.makedirs("data/simulation", exist_ok=True)
    df.to_csv(SUMMARY_CSV, index=False)
    print(f"Saved summary CSV: {SUMMARY_CSV}")


def parse_tripinfo():
    if not os.path.exists(TRIPINFO_XML):
        print(f"Tripinfo file not found: {TRIPINFO_XML}")
        return

    tree = ET.parse(TRIPINFO_XML)
    root = tree.getroot()

    rows = []

    for trip in root.findall("tripinfo"):
        rows.append({
            "vehicle_id": trip.attrib.get("id"),
            "depart": float(trip.attrib.get("depart", 0)),
            "arrival": float(trip.attrib.get("arrival", 0)),
            "duration": float(trip.attrib.get("duration", 0)),
            "routeLength": float(trip.attrib.get("routeLength", 0)),
            "waitingTime": float(trip.attrib.get("waitingTime", 0)),
            "waitingCount": int(trip.attrib.get("waitingCount", 0)),
            "timeLoss": float(trip.attrib.get("timeLoss", 0)),
            "departDelay": float(trip.attrib.get("departDelay", 0))
        })

    df = pd.DataFrame(rows)
    os.makedirs("data/simulation", exist_ok=True)
    df.to_csv(TRIPINFO_CSV, index=False)
    print(f"Saved tripinfo CSV: {TRIPINFO_CSV}")


def main():
    parse_summary()
    parse_tripinfo()


if __name__ == "__main__":
    main()
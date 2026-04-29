"""
scheduler.py
------------
Automatically fetches TomTom traffic data at a regular interval and
builds up the training dataset over time.

Usage:
    python scheduler.py                   # runs every 5 minutes (default)
    python scheduler.py --interval 10     # runs every 10 minutes
    python scheduler.py --count 50        # collect exactly 50 samples then stop
    python scheduler.py --interval 2 --count 100   # 100 samples, 2-min apart

Tip: Run this once before your demo/presentation to collect data overnight.
     Even 30-40 samples is enough to train the model.
     200+ gives a well-generalised model.
"""

import argparse
import time
import sys
import os
from datetime import datetime

# Make sure we can import from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from fetch_tomtom_data import fetch_tomtom, save_row, TOMTOM_FILE, TRAINING_FILE

# classify congestion from speed ratio
def _classify(speed_ratio: float) -> str:
    if speed_ratio < 0.4:
        return "High"
    elif speed_ratio < 0.7:
        return "Medium"
    else:
        return "Low"


def run_scheduler(interval_minutes: int = 5, max_count: int | None = None):
    interval_sec = interval_minutes * 60
    count = 0

    print("=" * 60)
    print("  TomTom Traffic Data Scheduler")
    print(f"  Fetch interval : every {interval_minutes} minute(s)")
    print(f"  Target samples : {'unlimited' if max_count is None else max_count}")
    print(f"  Started at     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("  Press Ctrl+C to stop.")
    print("=" * 60)

    while True:
        count += 1
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n[{ts}] Fetch #{count} …", end=" ", flush=True)

        row = fetch_tomtom()

        if row:
            # 1. Live display file
            save_row(row, TOMTOM_FILE)

            # 2. Training dataset
            training_row = {
                "currentSpeed":    row["currentSpeed"],
                "freeFlowSpeed":   row["freeFlowSpeed"],
                "speed_ratio":     row["speed_ratio"],
                "delay_seconds":   row["delay_seconds"],
                "roadClosure":     int(row["roadClosure"]),
                "congestion_label": _classify(row["speed_ratio"]),
            }
            save_row(training_row, TRAINING_FILE)

            print(
                f"✅  speed={row['currentSpeed']} km/h  "
                f"ratio={row['speed_ratio']:.2f}  "
                f"label={_classify(row['speed_ratio'])}"
            )
        else:
            print("❌  Fetch failed – will retry next cycle.")

        if max_count and count >= max_count:
            print(f"\n✅ Collected {count} samples. Scheduler finished.")
            break

        print(f"   Next fetch in {interval_minutes} minute(s) …")
        time.sleep(interval_sec)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TomTom data collection scheduler")
    parser.add_argument(
        "--interval", type=int, default=5,
        help="Minutes between fetches (default: 5)"
    )
    parser.add_argument(
        "--count", type=int, default=None,
        help="Stop after this many successful fetches (default: run forever)"
    )
    args = parser.parse_args()
    run_scheduler(interval_minutes=args.interval, max_count=args.count)
import schedule
import time
from src.data_collection.tomtom_collector import collect_all_locations, append_to_csv


def job():
    print("\nRunning scheduled traffic data collection...")
    data = collect_all_locations()
    append_to_csv(data)
    print("Collection cycle completed.\n")


# Run once immediately
job()

# Schedule every 5 minutes
schedule.every(5).minutes.do(job)

print("Scheduler started. Collecting traffic data every 5 minutes...")

while True:
    schedule.run_pending()
    time.sleep(1)
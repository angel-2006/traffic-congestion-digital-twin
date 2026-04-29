import pandas as pd

df = pd.read_csv("data/raw/tomtom_traffic_data.csv")

print("Total Rows:", len(df))
print("Unique Locations:", df["location_name"].nunique())
print("Rows Per Location:")
print(df["location_name"].value_counts())

print("\nTime Range:")
print("Start:", df["timestamp"].min())
print("End:", df["timestamp"].max())
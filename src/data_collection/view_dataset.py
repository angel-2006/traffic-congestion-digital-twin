import pandas as pd

df = pd.read_csv("data/raw/tomtom_traffic_data.csv")

print("Dataset Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 Rows:")
print(df.head())
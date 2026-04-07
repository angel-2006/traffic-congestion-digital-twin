import pandas as pd
import os

# -------------------------------
# FILE PATHS
# -------------------------------
input_file = "data/traffic_data.csv"
output_file = "data/future_traffic_data.csv"

# -------------------------------
# LOAD DATA
# -------------------------------
df = pd.read_csv(input_file)

print("Original data loaded successfully!")
print("Shape before processing:", df.shape)

# -------------------------------
# FUNCTION TO CREATE CONGESTION LABEL
# -------------------------------
def get_congestion_label(avg_speed):
    if avg_speed < 5:
        return "High"
    elif avg_speed < 15:
        return "Medium"
    else:
        return "Low"

# -------------------------------
# CREATE CURRENT CONGESTION LABEL
# -------------------------------
df["current_congestion"] = df["avg_speed"].apply(get_congestion_label)

# -------------------------------
# CREATE FUTURE CONGESTION LABEL
# -------------------------------
# Shift by 5 steps (you can think of this as "next few minutes")
future_steps = 5
df["future_congestion"] = df["current_congestion"].shift(-future_steps)

# -------------------------------
# REMOVE LAST ROWS WITH NO FUTURE LABEL
# -------------------------------
df = df.dropna().reset_index(drop=True)

print("Shape after adding future labels:", df.shape)

# -------------------------------
# SAVE OUTPUT
# -------------------------------
os.makedirs("data", exist_ok=True)
df.to_csv(output_file, index=False)

print("\nFuture dataset created successfully!")
print(f"Saved to: {output_file}")

print("\nFirst 5 rows:")
print(df.head())
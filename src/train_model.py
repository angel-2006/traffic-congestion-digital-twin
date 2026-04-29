"""
train_model.py
--------------
Trains the congestion classifier using REAL data collected from the TomTom API.

Features used (all sourced from TomTom API responses):
    currentSpeed, freeFlowSpeed, speed_ratio, delay_seconds, roadClosure

Target: congestion_label  (Low / Medium / High)

Run this script after you have collected enough rows via fetch_tomtom_data.py.
A minimum of ~30 rows is needed; 200+ rows gives reliable results.
"""

import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -------------------------------
# PATHS
# -------------------------------
TRAINING_FILE = "data/tomtom_training_data.csv"
MODEL_PATH    = "models/congestion_model.pkl"

os.makedirs("models", exist_ok=True)

# -------------------------------
# LOAD REAL DATA
# -------------------------------
if not os.path.exists(TRAINING_FILE):
    raise FileNotFoundError(
        f"Training file not found: {TRAINING_FILE}\n"
        "Run fetch_tomtom_data.py several times (or schedule it) to collect data first."
    )

df = pd.read_csv(TRAINING_FILE)
print(f"Loaded {len(df)} rows from {TRAINING_FILE}")

# Drop any incomplete rows
df.dropna(inplace=True)

if len(df) < 10:
    raise ValueError(
        f"Only {len(df)} complete rows found. Collect more data before training."
    )

print("\nLabel distribution:")
print(df["congestion_label"].value_counts())

# -------------------------------
# FEATURES & TARGET
# -------------------------------
FEATURES = ["currentSpeed", "freeFlowSpeed", "speed_ratio", "delay_seconds", "roadClosure"]
TARGET   = "congestion_label"

X = df[FEATURES]
y = df[TARGET]

# -------------------------------
# TRAIN / TEST SPLIT
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining samples : {len(X_train)}")
print(f"Testing samples  : {len(X_test)}")

# -------------------------------
# TRAIN
# -------------------------------
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# -------------------------------
# EVALUATE
# -------------------------------
y_pred   = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nAccuracy : {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# -------------------------------
# SAVE
# -------------------------------
joblib.dump({"model": model, "features": FEATURES}, MODEL_PATH)
print(f"\n✅ Model saved to: {MODEL_PATH}")
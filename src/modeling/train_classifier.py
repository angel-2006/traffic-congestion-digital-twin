import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from xgboost import XGBClassifier

# =========================
# Paths
# =========================
INPUT_PATH = "data/processed/final_traffic_dataset.csv"
MODEL_PATH = "models/congestion_classifier.pkl"
ENCODER_PATH = "models/location_encoder.pkl"
TARGET_ENCODER_PATH = "models/target_encoder.pkl"

os.makedirs("models", exist_ok=True)

# =========================
# Load dataset
# =========================
df = pd.read_csv(INPUT_PATH)

print("Dataset shape:", df.shape)
print(df.head())

# =========================
# Create congestion ratio
# =========================
# congestion_ratio = currentSpeed / freeFlowSpeed
# lower ratio = more congestion
df["congestion_ratio"] = df["currentSpeed"] / df["freeFlowSpeed"]

# Avoid divide-by-zero issues
df["congestion_ratio"] = df["congestion_ratio"].replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=["congestion_ratio"])

# =========================
# Create congestion labels
# =========================
def classify_congestion(ratio):
    if ratio >= 0.9:
        return "Low"
    elif ratio >= 0.7:
        return "Medium"
    else:
        return "High"

df["congestion_level"] = df["congestion_ratio"].apply(classify_congestion)

print("\nCongestion label distribution:")
print(df["congestion_level"].value_counts())

# =========================
# Convert timestamp features
# =========================
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["hour"] = df["timestamp"].dt.hour
df["minute"] = df["timestamp"].dt.minute
df["dayofweek"] = df["timestamp"].dt.dayofweek

# =========================
# Encode location
# =========================
location_encoder = LabelEncoder()
df["location_encoded"] = location_encoder.fit_transform(df["location_name"])

# =========================
# Encode target
# =========================
target_encoder = LabelEncoder()
df["target"] = target_encoder.fit_transform(df["congestion_level"])

print("\nTarget classes:")
print(dict(zip(target_encoder.classes_, target_encoder.transform(target_encoder.classes_))))

# =========================
# Select features
# =========================
feature_cols = [
    "currentSpeed",
    "freeFlowSpeed",
    "currentTravelTime",
    "freeFlowTravelTime",
    "confidence",
    "roadClosure",
    "hour",
    "minute",
    "dayofweek",
    "location_encoded",
    "sim_running_avg",
    "sim_waiting_avg",
    "sim_meanSpeed_avg",
    "sim_meanTravelTime_avg"
]

# Keep only available columns
feature_cols = [col for col in feature_cols if col in df.columns]

X = df[feature_cols].copy()
y = df["target"]

# Convert boolean to int if needed
for col in X.columns:
    if X[col].dtype == bool:
        X[col] = X[col].astype(int)

# Fill missing values if any
X = X.fillna(0)

print("\nFeatures used:")
print(feature_cols)

# =========================
# Train-test split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# Train model
# =========================
model = XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    objective="multi:softmax",
    num_class=len(target_encoder.classes_),
    random_state=42,
    eval_metric="mlogloss"
)

model.fit(X_train, y_train)

# =========================
# Predictions
# =========================
y_pred = model.predict(X_test)

# =========================
# Evaluation
# =========================
acc = accuracy_score(y_test, y_pred)

print("\n===== CLASSIFICATION EVALUATION =====")
print(f"Accuracy: {acc:.4f}")

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=target_encoder.classes_
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# =========================
# Save artifacts
# =========================
joblib.dump(model, MODEL_PATH)
joblib.dump(location_encoder, ENCODER_PATH)
joblib.dump(target_encoder, TARGET_ENCODER_PATH)

print(f"\nModel saved to: {MODEL_PATH}")
print(f"Location encoder saved to: {ENCODER_PATH}")
print(f"Target encoder saved to: {TARGET_ENCODER_PATH}")
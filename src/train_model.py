import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -------------------------------
# Paths
# -------------------------------
DATA_PATH = "data/traffic_data.csv"
MODEL_PATH = "models/congestion_model.pkl"

# -------------------------------
# Load dataset
# -------------------------------
df = pd.read_csv(DATA_PATH)

print("First 5 rows of dataset:")
print(df.head())

print("\nDataset shape:", df.shape)
print("\nLabel counts:")
print(df["congestion_label"].value_counts())

# -------------------------------
# Features and target
# -------------------------------
X = df[[
    "vehicle_count",
    "avg_speed",
    "total_waiting_time",
    "stopped_vehicles"
]]

y = df["congestion_label"]

# -------------------------------
# Train-test split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# -------------------------------
# Train model
# -------------------------------
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# -------------------------------
# Predictions
# -------------------------------
y_pred = model.predict(X_test)

# -------------------------------
# Evaluation
# -------------------------------
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# -------------------------------
# Save model
# -------------------------------
os.makedirs("models", exist_ok=True)
joblib.dump(model, MODEL_PATH)

print(f"\nModel saved successfully to: {MODEL_PATH}")
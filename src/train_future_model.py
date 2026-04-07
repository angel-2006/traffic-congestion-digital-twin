import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -------------------------------
# FILE PATHS
# -------------------------------
input_file = "data/future_traffic_data.csv"
model_output = "models/future_congestion_model.pkl"

# -------------------------------
# LOAD DATA
# -------------------------------
df = pd.read_csv(input_file)

print("First 5 rows of future dataset:")
print(df.head())

print("\nDataset shape:", df.shape)

# -------------------------------
# SELECT FEATURES
# -------------------------------
X = df[["vehicle_count", "avg_speed", "total_waiting_time", "stopped_vehicles"]]
y = df["future_congestion"]

print("\nFeature sample:")
print(X.head())

print("\nLabel counts:")
print(y.value_counts())

# -------------------------------
# TRAIN / TEST SPLIT
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# -------------------------------
# TRAIN MODEL
# -------------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# -------------------------------
# EVALUATE MODEL
# -------------------------------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("\nFuture Congestion Model Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# -------------------------------
# SAVE MODEL
# -------------------------------
os.makedirs("models", exist_ok=True)
joblib.dump(model, model_output)

print(f"\nFuture model saved successfully to: {model_output}")
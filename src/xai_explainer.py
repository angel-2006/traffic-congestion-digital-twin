"""
xai_explainer.py
----------------
Explainable AI (XAI) for the Traffic Congestion Prediction model.

Uses SHAP (SHapley Additive exPlanations) to answer:
  "Which features does the model rely on most, and by how much?"

Two public functions are meant to be called from app.py:

    plot_global_importance(model, X_background)
        → bar chart of mean |SHAP| per feature  (overall model behaviour)

    explain_single_prediction(model, row_dict, feature_names)
        → waterfall / force plot for ONE live TomTom reading
          (why did the model predict "High" / "Medium" / "Low" RIGHT NOW?)

Install dependency:  pip install shap matplotlib
"""

import io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless – no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib
import shap


# -----------------------------------------------------------------------
# INTERNAL HELPERS
# -----------------------------------------------------------------------

def _load_model_and_features(model_path: str = "models/congestion_model.pkl"):
    payload = joblib.load(model_path)
    if isinstance(payload, dict):
        return payload["model"], payload["features"]
    # Legacy format: bare sklearn model
    FEATURES = ["currentSpeed", "freeFlowSpeed", "speed_ratio", "delay_seconds", "roadClosure"]
    return payload, FEATURES


def _build_explainer(model, X_background: pd.DataFrame):
    """Build a TreeExplainer (fast, exact for RandomForest/GBM)."""
    return shap.TreeExplainer(model, X_background)


# -----------------------------------------------------------------------
# 1.  GLOBAL FEATURE IMPORTANCE  (mean |SHAP| across dataset)
# -----------------------------------------------------------------------

def plot_global_importance(
    model_path: str = "models/congestion_model.pkl",
    training_csv: str = "data/tomtom_training_data.csv",
) -> bytes:
    """
    Returns a PNG (bytes) of the global SHAP feature importance bar chart.
    """
    model, features = _load_model_and_features(model_path)
    df = pd.read_csv(training_csv).dropna()
    X  = df[features]

    explainer   = _build_explainer(model, X)
    shap_values = explainer.shap_values(X)  # list of arrays [class0, class1, class2]

    # Mean absolute SHAP across all classes and samples
    if isinstance(shap_values, list):
        mean_abs = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    else:
        mean_abs = np.abs(shap_values).mean(axis=0)

    sorted_idx = np.argsort(mean_abs)[::-1]
    sorted_features = [features[i] for i in sorted_idx]
    sorted_vals     = mean_abs[sorted_idx]

    # Friendly labels
    label_map = {
        "currentSpeed":  "Current Speed (km/h)",
        "freeFlowSpeed": "Free Flow Speed (km/h)",
        "speed_ratio":   "Speed Ratio",
        "delay_seconds": "Delay (seconds)",
        "roadClosure":   "Road Closure",
    }
    labels = [label_map.get(f, f) for f in sorted_features]

    # Colour bars by rank
    palette = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4", "#9467bd"]
    colors  = [palette[i % len(palette)] for i in range(len(labels))]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(labels[::-1], sorted_vals[::-1], color=colors[::-1], edgecolor="white")
    ax.set_xlabel("Mean |SHAP Value|  (average impact on model output)", fontsize=10)
    ax.set_title("🔍 Global Feature Importance (SHAP)", fontsize=13, fontweight="bold")
    ax.bar_label(bars, fmt="%.4f", padding=4, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# -----------------------------------------------------------------------
# 2.  SINGLE PREDICTION EXPLANATION  (why this prediction for this row?)
# -----------------------------------------------------------------------

def explain_single_prediction(
    row_dict: dict,
    model_path: str = "models/congestion_model.pkl",
    training_csv: str = "data/tomtom_training_data.csv",
) -> bytes:
    """
    Returns a PNG (bytes) of a horizontal waterfall / contribution bar
    showing how each feature pushed the prediction toward the final class.

    row_dict  – the live TomTom feature dict (same keys as MODEL_FEATURES).
    """
    model, features = _load_model_and_features(model_path)
    df_bg = pd.read_csv(training_csv).dropna()
    X_bg  = df_bg[features]

    explainer = _build_explainer(model, X_bg)

    X_sample = pd.DataFrame([[row_dict[f] for f in features]], columns=features)
    shap_vals = explainer.shap_values(X_sample)  # list: [class0_arr, class1_arr, class2_arr]

    # Identify the predicted class index
    pred_class     = model.predict(X_sample)[0]
    class_names    = model.classes_.tolist()
    pred_class_idx = class_names.index(pred_class) if pred_class in class_names else 0
    sv_for_class   = shap_vals[pred_class_idx][0]   # shape (n_features,)

    label_map = {
        "currentSpeed":  "Current Speed",
        "freeFlowSpeed": "Free Flow Speed",
        "speed_ratio":   "Speed Ratio",
        "delay_seconds": "Delay (s)",
        "roadClosure":   "Road Closure",
    }
    labels = [label_map.get(f, f) for f in features]
    values = [row_dict[f] for f in features]

    # Sort by absolute contribution
    order       = np.argsort(np.abs(sv_for_class))[::-1]
    sv_sorted   = sv_for_class[order]
    lbl_sorted  = [f"{labels[i]}\n= {values[i]}" for i in order]

    colors = ["#d62728" if v > 0 else "#1f77b4" for v in sv_sorted]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(lbl_sorted[::-1], sv_sorted[::-1], color=colors[::-1], edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP Value  (positive = pushes toward predicted class)", fontsize=9)
    ax.set_title(
        f"🔍 Why did the model predict \"{pred_class}\"?\n"
        f"Red = increases prediction  |  Blue = decreases prediction",
        fontsize=11, fontweight="bold"
    )
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# -----------------------------------------------------------------------
# 3.  FEATURE CORRELATION TABLE  (human-readable sanity check)
# -----------------------------------------------------------------------

def feature_stats_table(
    training_csv: str = "data/tomtom_training_data.csv",
) -> pd.DataFrame:
    """
    Returns a DataFrame with per-feature mean values split by congestion label.
    Lets you quickly sanity-check: 'High' congestion should show low speed_ratio etc.
    """
    df = pd.read_csv(training_csv).dropna()
    if "congestion_label" not in df.columns:
        return pd.DataFrame()
    numeric_cols = ["currentSpeed", "freeFlowSpeed", "speed_ratio", "delay_seconds"]
    available    = [c for c in numeric_cols if c in df.columns]
    return df.groupby("congestion_label")[available].mean().round(3)


# -----------------------------------------------------------------------
# CLI  (run standalone to generate & save plots)
# -----------------------------------------------------------------------

if __name__ == "__main__":
    import os, sys

    os.makedirs("outputs", exist_ok=True)

    print("Generating global SHAP importance chart …")
    img = plot_global_importance()
    with open("outputs/shap_global_importance.png", "wb") as f:
        f.write(img)
    print("  Saved → outputs/shap_global_importance.png")

    print("\nFeature averages by congestion label:")
    print(feature_stats_table())
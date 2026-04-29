"""
app.py  –  Traffic Congestion Prediction using Digital Twin & ML
================================================================
Changes in this version:
1. Model trained on REAL TomTom data  → features match TomTom API fields.
2. Prediction uses the LIVE TomTom fetch.
3. "Launch SUMO Digital Twin" opens sumo-gui with red/orange/green road colors.
4. Scheduler: collect training data automatically at a configurable interval.
5. XAI tab: SHAP global feature importance + per-prediction explanation.
"""

import os
import sys
import time
import subprocess
import threading

import joblib
import pandas as pd
import requests
import streamlit as st
from datetime import datetime

# Make src/ importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# -----------------------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------------------
st.set_page_config(
    page_title="Traffic Digital Twin – Palace Grounds",
    page_icon="🚦",
    layout="wide",
)

# -----------------------------------------------------------------------
# CONSTANTS
# -----------------------------------------------------------------------
TOMTOM_API_KEY  = "x2xyWWG4o9tNtVizpeLzpwR2GN20Y2uW"   # <-- replace with your key
LAT             = 12.9986
LON             = 77.5926
TOMTOM_URL      = (
    f"https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"
    f"?point={LAT},{LON}&key={TOMTOM_API_KEY}"
)

MODEL_FILE      = "models/congestion_model.pkl"
TOMTOM_CSV      = "data/tomtom_traffic.csv"
TRAINING_CSV    = "data/tomtom_training_data.csv"
MODEL_FEATURES  = ["currentSpeed", "freeFlowSpeed", "speed_ratio", "delay_seconds", "roadClosure"]

# -----------------------------------------------------------------------
# SESSION STATE
# -----------------------------------------------------------------------
if "scheduler_running" not in st.session_state:
    st.session_state.scheduler_running = False
if "scheduler_stop" not in st.session_state:
    st.session_state.scheduler_stop = threading.Event()

# -----------------------------------------------------------------------
# LOAD MODEL
# -----------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading ML model…")
def load_model():
    if not os.path.exists(MODEL_FILE):
        return None, None
    payload = joblib.load(MODEL_FILE)
    if isinstance(payload, dict):
        return payload["model"], payload.get("features", MODEL_FEATURES)
    return payload, MODEL_FEATURES

model, model_features = load_model()

# -----------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------
def fetch_live_tomtom():
    try:
        resp = requests.get(TOMTOM_URL, timeout=10)
        if resp.status_code == 200:
            flow      = resp.json()["flowSegmentData"]
            current   = float(flow["currentSpeed"])
            free_flow = float(flow["freeFlowSpeed"])
            curr_tt   = float(flow["currentTravelTime"])
            free_tt   = float(flow["freeFlowTravelTime"])
            closure   = int(flow["roadClosure"])
            ratio     = current / free_flow if free_flow > 0 else 0.0
            return {
                "timestamp":          datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "currentSpeed":       current,
                "freeFlowSpeed":      free_flow,
                "currentTravelTime":  curr_tt,
                "freeFlowTravelTime": free_tt,
                "roadClosure":        closure,
                "speed_ratio":        round(ratio, 4),
                "delay_seconds":      round(curr_tt - free_tt, 2),
            }
    except Exception:
        pass
    return None

def append_to_csv(row, filepath):
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
    new_df = pd.DataFrame([row])
    if os.path.exists(filepath):
        combined = pd.concat([pd.read_csv(filepath), new_df], ignore_index=True)
        combined.to_csv(filepath, index=False)
    else:
        new_df.to_csv(filepath, index=False)

def classify_tomtom(speed_ratio):
    if speed_ratio < 0.4:   return "High"
    elif speed_ratio < 0.7: return "Medium"
    else:                   return "Low"

def predict_congestion(row):
    if model is None:
        return "N/A (no model)"
    try:
        feat = model_features or MODEL_FEATURES
        X    = pd.DataFrame([[row[f] for f in feat]], columns=feat)
        return model.predict(X)[0]
    except Exception as e:
        return f"Error: {e}"

def get_suggestions(level):
    return {
        "Low":    ["✅ Traffic is flowing smoothly.", "✅ No action needed."],
        "Medium": ["⚠️ Moderate congestion detected.",
                   "⚠️ Monitor traffic inflow near junction.",
                   "⚠️ Prepare alternate route advisories."],
        "High":   ["🚨 Heavy congestion – act immediately!",
                   "🚨 Divert traffic via alternate roads.",
                   "🚨 Issue congestion alert to public.",
                   "🚨 Notify traffic control personnel."],
    }.get(level, ["ℹ️ No suggestions available."])

def congestion_color(level):
    return {"Low": "🟢", "Medium": "🟠", "High": "🔴"}.get(level, "⚪")

def ensure_derived_fields(row):
    if "speed_ratio" not in row:
        cs = float(row.get("currentSpeed", 1))
        ff = float(row.get("freeFlowSpeed", 1))
        row["speed_ratio"]   = cs / ff if ff > 0 else 0.0
        row["delay_seconds"] = float(row.get("currentTravelTime", 0)) - float(row.get("freeFlowTravelTime", 0))
    if "roadClosure" not in row:
        row["roadClosure"] = 0
    return row

# -----------------------------------------------------------------------
# SCHEDULER (background thread)
# -----------------------------------------------------------------------
def _scheduler_worker(interval_minutes, stop_event):
    interval_sec = interval_minutes * 60
    while not stop_event.is_set():
        row = fetch_live_tomtom()
        if row:
            append_to_csv(row, TOMTOM_CSV)
            training_row = {
                "currentSpeed":     row["currentSpeed"],
                "freeFlowSpeed":    row["freeFlowSpeed"],
                "speed_ratio":      row["speed_ratio"],
                "delay_seconds":    row["delay_seconds"],
                "roadClosure":      int(row["roadClosure"]),
                "congestion_label": classify_tomtom(row["speed_ratio"]),
            }
            append_to_csv(training_row, TRAINING_CSV)
        stop_event.wait(interval_sec)

def start_scheduler(interval_minutes):
    st.session_state.scheduler_stop.clear()
    t = threading.Thread(
        target=_scheduler_worker,
        args=(interval_minutes, st.session_state.scheduler_stop),
        daemon=True,
    )
    t.start()
    st.session_state.scheduler_running = True

def stop_scheduler():
    st.session_state.scheduler_stop.set()
    st.session_state.scheduler_running = False

# -----------------------------------------------------------------------
# HEADER
# -----------------------------------------------------------------------
st.title("🚦 Traffic Congestion Prediction using Digital Twin & ML")
st.subheader("📍 Palace Grounds Junction, Bengaluru")
st.caption(f"Page loaded: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
st.markdown("---")

# -----------------------------------------------------------------------
# TABS
# -----------------------------------------------------------------------
tab_dashboard, tab_scheduler, tab_xai = st.tabs([
    "📊 Dashboard",
    "⏱️ Data Scheduler",
    "🔍 XAI – Explainability",
])

# ═══════════════════════════════════════════════════════════════════════
# TAB 1 – DASHBOARD
# ═══════════════════════════════════════════════════════════════════════
with tab_dashboard:

    st.markdown("### ⚙️ Control Panel")
    c1, c2, c3, c4 = st.columns(4)
    with c1:  fetch_btn   = st.button("🔄 Fetch & Save TomTom Data")
    with c2:  launch_btn  = st.button("🚀 Launch SUMO Digital Twin")
    with c3:  train_btn   = st.button("🧠 Retrain Model on Real Data")
    with c4:  refresh_btn = st.button("🔁 Refresh Dashboard")

    if fetch_btn:
        with st.spinner("Fetching live TomTom data…"):
            row = fetch_live_tomtom()
        if row:
            append_to_csv(row, TOMTOM_CSV)
            tr = {k: row[k] for k in MODEL_FEATURES if k in row}
            tr["congestion_label"] = classify_tomtom(row["speed_ratio"])
            append_to_csv(tr, TRAINING_CSV)
            st.success(f"✅ Saved  |  speed={row['currentSpeed']} km/h  |  ratio={row['speed_ratio']}")
        else:
            st.error("❌ Failed to fetch TomTom data.")
        st.rerun()

    if launch_btn:
        threading.Thread(
            target=lambda: subprocess.Popen(["python", "src/control_sumo_live.py"]),
            daemon=True,
        ).start()
        st.success("🚀 SUMO-GUI launched! Roads will be coloured 🔴/🟠/🟢 using live TomTom data.")

    if train_btn:
        with st.spinner("Retraining model on real TomTom data…"):
            result = subprocess.run(["python", "src/train_model.py"], capture_output=True, text=True)
        if result.returncode == 0:
            st.success("✅ Model retrained! Reload the page to apply it.")
            st.code(result.stdout[-2000:])
        else:
            st.error("❌ Training failed.")
            st.code(result.stderr[-2000:])

    if refresh_btn:
        st.rerun()

    st.markdown("---")

    with st.spinner("Fetching live TomTom data…"):
        live = fetch_live_tomtom()

    if live is None:
        if os.path.exists(TOMTOM_CSV):
            df_saved = pd.read_csv(TOMTOM_CSV)
            live_row = ensure_derived_fields(df_saved.iloc[-1].to_dict()) if not df_saved.empty else None
            if live_row:
                st.warning("⚠️ Using last saved TomTom reading (API unreachable).")
            else:
                st.error("No data available."); st.stop()
        else:
            st.error("No data available. Fetch TomTom data first."); st.stop()
    else:
        live_row = ensure_derived_fields(live)

    real_state = classify_tomtom(live_row["speed_ratio"])
    predicted  = predict_congestion(live_row)

    st.markdown("### 📊 Live Traffic Overview")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Current Speed",   f"{live_row['currentSpeed']} km/h")
    m2.metric("Free Flow Speed", f"{live_row['freeFlowSpeed']} km/h")
    m3.metric("Speed Ratio",     f"{live_row['speed_ratio']:.2f}")
    m4.metric("Real-Time State", f"{congestion_color(real_state)} {real_state}")
    m5.metric("ML Prediction",   f"{congestion_color(predicted)} {predicted}")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🌍 Real-Time TomTom Data")
        st.write(f"**Timestamp:** {live_row.get('timestamp', 'N/A')}")
        st.write(f"**Current Speed:** {live_row['currentSpeed']} km/h")
        st.write(f"**Free Flow Speed:** {live_row['freeFlowSpeed']} km/h")
        st.write(f"**Speed Ratio:** {live_row['speed_ratio']:.2f}")
        st.write(f"**Delay:** {live_row.get('delay_seconds', 0):.1f} s")
        st.write(f"**Road Closure:** {'Yes' if live_row.get('roadClosure') else 'No'}")
        st.write(f"**State:** {congestion_color(real_state)} **{real_state}**")
    with col2:
        st.markdown("#### 🔮 ML Prediction")
        if model:
            lc = {"Low": "green", "Medium": "orange", "High": "red"}.get(predicted, "gray")
            st.markdown(f"<h2 style='color:{lc}'>{congestion_color(predicted)} {predicted}</h2>", unsafe_allow_html=True)
        else:
            st.warning("Model not found. Collect data then click Retrain Model.")

    st.markdown("---")
    st.markdown("### 🛣️ Traffic Management Suggestions")
    for s in get_suggestions(predicted):
        st.write(s)

    st.markdown("---")
    st.markdown("### 🎛️ What-If Analysis")
    w1, w2, w3 = st.columns(3)
    with w1:
        wi_current = st.slider("Current Speed (km/h)", 0, 120, int(live_row["currentSpeed"]))
        wi_free    = st.slider("Free Flow Speed (km/h)", 1, 120, int(live_row["freeFlowSpeed"]))
    with w2:
        wi_delay   = st.slider("Delay (seconds)", 0, 600, int(max(0, live_row.get("delay_seconds", 0))))
        wi_closure = st.selectbox("Road Closure", ["No", "Yes"])
    with w3:
        wi_ratio = wi_current / wi_free if wi_free > 0 else 0.0
        st.metric("Simulated Speed Ratio", f"{wi_ratio:.2f}")
        st.metric("Rule-Based State", f"{congestion_color(classify_tomtom(wi_ratio))} {classify_tomtom(wi_ratio)}")

    wi_row  = {"currentSpeed": wi_current, "freeFlowSpeed": wi_free, "speed_ratio": wi_ratio,
               "delay_seconds": wi_delay, "roadClosure": 1 if wi_closure == "Yes" else 0}
    st.warning(f"🔮 What-If Prediction: **{congestion_color(predict_congestion(wi_row))} {predict_congestion(wi_row)}**")

    st.markdown("---")
    with st.expander("📂 Training Data & History"):
        if os.path.exists(TRAINING_CSV):
            df_t = pd.read_csv(TRAINING_CSV)
            st.write(f"**{len(df_t)} training samples collected.**")
            st.dataframe(df_t.tail(20))
            if "congestion_label" in df_t.columns:
                st.bar_chart(df_t["congestion_label"].value_counts())
        else:
            st.info("No training data yet.")

    st.caption("Dashboard auto-refreshes every 60 seconds.")
    time.sleep(60)
    st.rerun()


# ═══════════════════════════════════════════════════════════════════════
# TAB 2 – SCHEDULER
# ═══════════════════════════════════════════════════════════════════════
with tab_scheduler:
    st.markdown("### ⏱️ Automatic Data Collection Scheduler")
    st.write(
        "Runs in the background and fetches TomTom traffic data at a set interval, "
        "automatically building up your real training dataset."
    )

    interval = st.slider("Fetch interval (minutes)", min_value=1, max_value=60, value=5, step=1)

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("▶️ Start Scheduler", disabled=st.session_state.scheduler_running):
            start_scheduler(interval)
            st.success(f"✅ Scheduler started – fetching every {interval} minute(s).")
    with col_b:
        if st.button("⏹️ Stop Scheduler", disabled=not st.session_state.scheduler_running):
            stop_scheduler()
            st.info("Scheduler stopped.")

    st.metric("Scheduler Status",
              "🟢 Running" if st.session_state.scheduler_running else "🔴 Stopped")

    st.markdown("---")
    st.markdown("#### 📋 Training Data Progress")

    if os.path.exists(TRAINING_CSV):
        df_prog = pd.read_csv(TRAINING_CSV)
        n = len(df_prog)
        st.write(f"**{n} samples collected so far.**")
        st.progress(min(n / 200, 1.0), text=f"{n} / 200 recommended samples")
        if "congestion_label" in df_prog.columns:
            st.bar_chart(df_prog["congestion_label"].value_counts())
        st.dataframe(df_prog.tail(10))
    else:
        st.info("No training data yet. Start the scheduler or manually fetch data.")

    st.markdown("---")
    st.markdown(
        "**Run from terminal for overnight collection:**\n"
        "```bash\npython scheduler.py --interval 5 --count 200\n```"
    )


# ═══════════════════════════════════════════════════════════════════════
# TAB 3 – XAI
# ═══════════════════════════════════════════════════════════════════════
with tab_xai:
    st.markdown("### 🔍 Explainable AI (XAI) – How does the model make decisions?")
    st.write(
        "Uses **SHAP (SHapley Additive exPlanations)** to answer: "
        "*Which features does the model actually rely on, and by how much?*"
    )

    if model is None:
        st.error("No trained model found. Train the model first (Dashboard → Retrain Model).")
        st.stop()

    if not os.path.exists(TRAINING_CSV):
        st.warning("No training data found. Collect data before using XAI.")
        st.stop()

    df_check = pd.read_csv(TRAINING_CSV)
    if len(df_check) < 5:
        st.warning(f"Only {len(df_check)} samples – need at least 5 for SHAP. Collect more data.")
        st.stop()

    try:
        from xai_explainer import (
            plot_global_importance,
            explain_single_prediction,
            feature_stats_table,
        )
        xai_ok = True
    except ImportError:
        xai_ok = False
        st.error("Install SHAP first:\n```\npip install shap matplotlib\n```")

    if xai_ok:

        # ---- 1. Global importance ----
        st.markdown("#### 📊 Global Feature Importance")
        st.write(
            "Average impact of each feature across **all training samples**. "
            "The model is behaving correctly if `speed_ratio` and `delay_seconds` "
            "rank at the top — they are the strongest real-world signals of congestion."
        )
        with st.spinner("Computing SHAP values across training data…"):
            try:
                st.image(plot_global_importance(MODEL_FILE, TRAINING_CSV), use_column_width=True)
            except Exception as e:
                st.error(f"Global SHAP error: {e}")

        st.markdown("---")

        # ---- 2. Feature stats table ----
        st.markdown("#### 📋 Feature Averages by Congestion Level")
        st.write(
            "Sanity check: **High** congestion rows should have **low speed ratio** "
            "and **high delay**. If that pattern holds, the model is learning the right signal."
        )
        try:
            stats_df = feature_stats_table(TRAINING_CSV)
            if not stats_df.empty:
                st.dataframe(stats_df.style.background_gradient(axis=0, cmap="RdYlGn_r"))
        except Exception as e:
            st.warning(f"Stats table error: {e}")

        st.markdown("---")

        # ---- 3. Live prediction explanation ----
        st.markdown("#### 🎯 Explain the Current Live Prediction")
        st.write(
            "**Red bars** push the prediction *toward* the predicted class. "
            "**Blue bars** push it *away*."
        )
        live_xai = fetch_live_tomtom()
        if live_xai is None and os.path.exists(TOMTOM_CSV):
            df_s = pd.read_csv(TOMTOM_CSV)
            live_xai = df_s.iloc[-1].to_dict() if not df_s.empty else None

        if live_xai:
            live_xai = ensure_derived_fields(live_xai)
            pred_xai = predict_congestion(live_xai)
            st.write(f"**Live prediction:** {congestion_color(pred_xai)} **{pred_xai}**")
            with st.spinner("Computing SHAP for live reading…"):
                try:
                    st.image(explain_single_prediction(live_xai, MODEL_FILE, TRAINING_CSV), use_column_width=True)
                except Exception as e:
                    st.error(f"Single-prediction SHAP error: {e}")
        else:
            st.info("No live data available.")

        st.markdown("---")

        # ---- 4. Custom scenario ----
        st.markdown("#### 🛠️ Explain a Custom Scenario")
        cc1, cc2, cc3, cc4 = st.columns(4)
        with cc1: ex_current = st.number_input("Current Speed", 0, 120, 40, key="ex_cs")
        with cc2: ex_free    = st.number_input("Free Flow Speed", 1, 120, 80, key="ex_ff")
        with cc3: ex_delay   = st.number_input("Delay (seconds)", 0, 600, 60, key="ex_dl")
        with cc4: ex_closure = st.selectbox("Road Closure", ["No", "Yes"], key="ex_cl")

        if st.button("🔍 Explain This Scenario"):
            ex_ratio = ex_current / ex_free if ex_free > 0 else 0.0
            ex_row   = {
                "currentSpeed":  ex_current,
                "freeFlowSpeed": ex_free,
                "speed_ratio":   ex_ratio,
                "delay_seconds": ex_delay,
                "roadClosure":   1 if ex_closure == "Yes" else 0,
            }
            ex_pred = predict_congestion(ex_row)
            st.write(f"**Predicted:** {congestion_color(ex_pred)} **{ex_pred}**  |  Speed ratio: {ex_ratio:.2f}")
            with st.spinner("Computing SHAP…"):
                try:
                    st.image(explain_single_prediction(ex_row, MODEL_FILE, TRAINING_CSV), use_column_width=True)
                except Exception as e:
                    st.error(f"SHAP error: {e}")
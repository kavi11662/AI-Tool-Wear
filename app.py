import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="AI Tool Wear Monitor", page_icon="⚙️", layout="wide")

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
body { background-color:#f5f7fb; font-family:sans-serif; }

/* Headers */
.main-title { font-size:42px; font-weight:800; color:#2563eb; text-align:center; margin-bottom:5px; }
.sub-text { font-size:18px; color:#64748b; text-align:center; margin-bottom:20px; }

/* Section Header */
.section-header { font-size:22px; font-weight:700; color:#fff; background:linear-gradient(90deg,#2563eb,#06b6d4); padding:8px; border-radius:8px; margin-bottom:12px; }

/* Metric Cards */
.metric-card { padding:15px; border-radius:12px; background:linear-gradient(90deg,#06b6d4,#2563eb); color:white; text-align:center; margin-bottom:15px; font-weight:bold; }

/* Status Colors */
.status-green {color:#16a34a;font-weight:600;font-size:18px;}
.status-yellow {color:#f59e0b;font-weight:600;font-size:18px;}
.status-red {color:#dc2626;font-weight:600;font-size:18px;}

/* Footer */
.footer { text-align:center; color:#64748b; font-size:14px; margin-top:30px; }
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown('<p class="main-title">🔧 AI Tool Wear Monitoring Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-text">Smart Manufacturing • Predictive Maintenance • Industry 4.0</p>', unsafe_allow_html=True)
st.markdown("---")

# ---------------- LOAD MODEL ----------------
model = joblib.load("tool_wear_xgb_pipeline.pkl")

# ---------------- SIDEBAR INPUTS ----------------
st.sidebar.header("⚙️ Machine Parameters")
case = st.sidebar.selectbox("Case", [1, 2, 3])
run = st.sidebar.text_input("Run Number", "1")
DOC = st.sidebar.text_input("Depth of Cut", "1.5")
feed = st.sidebar.text_input("Feed Rate", "0.5")
material = st.sidebar.selectbox("Material", [1, 2])
smcAC = st.sidebar.text_input("Motor Current AC", "0.2")
smcDC = st.sidebar.text_input("Motor Current DC", "0.2")
vib_table = st.sidebar.text_input("Table Vibration", "0.1")
vib_spindle = st.sidebar.text_input("Spindle Vibration", "0.1")
AE_table = st.sidebar.text_input("Acoustic Emission Table", "0.1")
AE_spindle = st.sidebar.text_input("Acoustic Emission Spindle", "0.1")

# Convert text inputs to float/int
try:
    run = int(run)
    DOC = float(DOC)
    feed = float(feed)
    smcAC = float(smcAC)
    smcDC = float(smcDC)
    vib_table = float(vib_table)
    vib_spindle = float(vib_spindle)
    AE_table = float(AE_table)
    AE_spindle = float(AE_spindle)
except:
    st.sidebar.error("Please enter valid numeric values for all parameters.")

# ---------------- PREPARE INPUT DATA ----------------
input_dict = {
    "case":[case], "run":[run], "DOC":[DOC], "feed":[feed], "material":[material],
    "smcAC":[smcAC], "smcDC":[smcDC], "vib_table":[vib_table], "vib_spindle":[vib_spindle],
    "AE_table":[AE_table], "AE_spindle":[AE_spindle]
}
input_df = pd.DataFrame(input_dict)
input_df = pd.get_dummies(input_df)
input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)

# ---------------- SESSION STATE FOR REPORTS ----------------
if 'report_history' not in st.session_state:
    st.session_state['report_history'] = pd.DataFrame(columns=[
        "Case","Run","Material","DOC","Feed","smcAC","smcDC",
        "Vib Table","Vib Spindle","AE Table","AE Spindle",
        "Predicted Wear","Remaining Life (%)","Health Score"
    ])

# ---------------- PREDICTION ----------------
if st.button("🚀 Run Smart Diagnosis"):
    wear = float(model.predict(input_df)[0])
    wear = max(wear,0)
    failure_threshold = 0.7
    rul = max(failure_threshold - wear, 0)
    rul_percent = (rul / failure_threshold) * 100
    health_score = max(100 - (wear/failure_threshold)*100, 0)
    avg_wear_rate = 0.02
    remaining_cycles = int(rul / avg_wear_rate) if avg_wear_rate>0 else 0

    # --- KPI CARDS ---
    st.markdown('<div class="section-header">📊 Prediction Results</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f'<div class="metric-card">🔧 Tool Wear: {wear:.3f} mm</div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card">⏳ Remaining Life: {rul_percent:.1f}%<br>🔁 Remaining Cycles: {remaining_cycles}</div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card">💚 Health Score: {health_score:.0f}/100</div>', unsafe_allow_html=True)

    # --- TOOL CONDITION ---
    st.markdown('<div class="section-header">🏷 Tool Condition</div>', unsafe_allow_html=True)
    if wear < 0.4:
        st.markdown('<p class="status-green">HEALTHY 🟢</p>', unsafe_allow_html=True)
    elif wear < 0.6:
        st.markdown('<p class="status-yellow">REPLACE SOON 🟡</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p class="status-red">REPLACE IMMEDIATELY 🔴</p>', unsafe_allow_html=True)

    # --- FUTURE PROJECTION ---
    st.markdown('<div class="section-header">📈 AI Future Wear Forecast</div>', unsafe_allow_html=True)
    future_cycles = np.arange(1,11)
    projected_wear = []
    current_wear = wear
    load_factor = (DOC*feed)+(smcAC+smcDC)+(vib_table+vib_spindle)+(AE_table+AE_spindle)
    base_increment = 0.003 + (load_factor*0.0015)
    for cycle in future_cycles:
        acceleration = 1 + (current_wear/failure_threshold)
        current_wear += base_increment*acceleration
        current_wear = min(current_wear,failure_threshold)
        projected_wear.append(current_wear)
    trend_df = pd.DataFrame({"Cycle":future_cycles, "Projected Wear":projected_wear})
    st.line_chart(trend_df.set_index("Cycle"))

    # --- FEATURE IMPORTANCE ---
    st.markdown('<div class="section-header">🧠 AI Decision Factors</div>', unsafe_allow_html=True)
    try:
        xgb_model = model.named_steps["model"]
        importance = xgb_model.feature_importances_
        features = model.feature_names_in_
        imp_df = pd.DataFrame({"Feature":features,"Importance":importance}).sort_values(by="Importance",ascending=False)
        st.bar_chart(imp_df.set_index("Feature"))
    except:
        st.info("Feature importance not available.")

    # --- ADD TO RECENT REPORTS ---
    new_report = pd.DataFrame([{
        "Case":case, "Run":run, "Material":material,
        "DOC":DOC, "Feed":feed, "smcAC":smcAC, "smcDC":smcDC,
        "Vib Table":vib_table, "Vib Spindle":vib_spindle,
        "AE Table":AE_table, "AE Spindle":AE_spindle,
        "Predicted Wear":round(wear,3),
        "Remaining Life (%)":round(rul_percent,1),
        "Health Score":round(health_score,0)
    }])
    st.session_state['report_history'] = pd.concat([new_report, st.session_state['report_history']], ignore_index=True)

# --- RECENT REPORTS TABLE ---
st.markdown('<div class="section-header">📋 Recent Reports</div>', unsafe_allow_html=True)
st.dataframe(st.session_state['report_history'].head(10))

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown('<div class="footer">AI Predictive Maintenance Dashboard | XGBoost Model | NASA Milling Dataset | R² = 0.69</div>', unsafe_allow_html=True)

import streamlit as st
import pandas as pd
from pathlib import Path
import joblib
from datetime import datetime
import os
from openai import OpenAI

st.set_page_config(
    page_title="Predictive Maintenance",
    page_icon="⚙️",
    layout="centered"
)

st.markdown("""
<style>
.stApp {
    background-color: #F5F9FC;
}

.block-container {
    max-width: 900px;
    padding-top: 35px;
}

.title {
    text-align: center;
    font-size: 38px;
    font-weight: 800;
    color: #173F5F;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    color: #66808C;
    font-size: 16px;
    margin-bottom: 30px;
}

.card {
    background-color: #FFFFFF;
    padding: 24px;
    border-radius: 16px;
    border: 1px solid #DCE8EF;
    box-shadow: 0 3px 12px rgba(23, 63, 95, 0.06);
    margin-bottom: 20px;
}

.section-title {
    color: #087E8B;
    font-size: 21px;
    font-weight: 700;
    margin-bottom: 18px;
}

label {
    color: #294C5A !important;
    font-weight: 600 !important;
}

/* WHITE INPUT BARS */
div[data-baseweb="input"] {
    background-color: #FFFFFF !important;
    border-radius: 8px !important;
}

div[data-baseweb="input"] > div {
    background-color: #FFFFFF !important;
}

div[data-baseweb="input"] input {
    background-color: #FFFFFF !important;
    color: #333333 !important;
    -webkit-text-fill-color: #333333 !important;
    font-weight: 600 !important;
}

/* WHITE SELECT BOXES */
div[data-baseweb="select"] {
    background-color: #FFFFFF !important;
    border-radius: 8px !important;
}

div[data-baseweb="select"] > div {
    background-color: #FFFFFF !important;
}

div[data-baseweb="select"] div {
    color: #333333 !important;
    font-weight: 600 !important;
}

div[data-baseweb="select"] span {
    color: #333333 !important;
    font-weight: 600 !important;
}

/* WHITE DATE INPUT */
div[data-baseweb="input"] input {
    background-color: #FFFFFF !important;
    color: #333333 !important;
    -webkit-text-fill-color: #333333 !important;
}

.stButton > button {
    width: 100%;
    height: 52px;
    background-color: #087E8B;
    color: white !important;
    border: none;
    border-radius: 10px;
    font-size: 17px;
    font-weight: 700;
}

.stButton > button:hover {
    background-color: #066873;
}

.low-risk {
    background-color: #EAF8F3;
    border: 2px solid #72C9B3;
    border-radius: 16px;
    padding: 28px;
    text-align: center;
    margin-top: 15px;
}

.low-risk h1 {
    color: #16826B;
    font-size: 30px;
    margin: 5px;
}

.low-risk h2 {
    color: #126653;
    font-size: 32px;
    margin: 8px;
}

.low-risk p {
    color: #49665F;
}

.high-risk {
    background-color: #FFF1F1;
    border: 2px solid #E99A9A;
    border-radius: 16px;
    padding: 28px;
    text-align: center;
    margin-top: 15px;
}

.high-risk h1 {
    color: #C0392B;
    font-size: 30px;
    margin: 5px;
}

.high-risk h2 {
    color: #A93226;
    font-size: 32px;
    margin: 8px;
}

.high-risk p {
    color: #704747;
}

[data-testid="stMetric"] {
    background-color: white;
    border: 1px solid #DCE8EF;
    border-radius: 12px;
    padding: 15px;
}

[data-testid="stMetricLabel"] {
    color: #333333 !important;
}

[data-testid="stMetricValue"] {
    color: #000000 !important;
}

[data-testid="stMetricValue"] div {
    color: #000000 !important;
}

/* ---------- AI Analysis ---------- */

.ai-card {
    background-color: #EFF6FF;
    border: 1px solid #93C5FD;
    border-radius: 14px;
    padding: 20px;
    margin-top: 15px;
}

.ai-title {
    color: #2563EB;
    font-size: 21px;
    font-weight: 700;
    margin-bottom: 10px;
}

.ai-text {
    color: #1F2937;
    font-size: 15px;
    line-height: 1.7;
    white-space: pre-wrap;
}

/* ---------- History ---------- */

.history-title {
    color: #6D28D9;
    font-size: 21px;
    font-weight: 700;
    margin-bottom: 15px;
}

.history-card {
    background-color: #F5F3FF;
    border: 1px solid #C4B5FD;
    border-radius: 14px;
    padding: 18px;
    margin-top: 12px;
}

.history-card h3 {
    color: #5B21B6;
    margin: 0 0 8px 0;
}

.history-card p {
    color: #374151;
    margin: 4px 0;
}

</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return joblib.load("model.pkl")


try:
    model = load_model()
except Exception as e:
    st.error("❌ Could not load model.pkl")
    st.code(str(e))
    st.stop()


HISTORY_FILE = "prediction_history.csv"


def save_prediction(record):
    new_record = pd.DataFrame([record])

    if Path(HISTORY_FILE).exists():
        old_history = pd.read_csv(HISTORY_FILE)
        history = pd.concat([old_history, new_record], ignore_index=True)
    else:
        history = new_record

    history.to_csv(HISTORY_FILE, index=False)


def load_history():
    if Path(HISTORY_FILE).exists():
        return pd.read_csv(HISTORY_FILE)
    return pd.DataFrame()



# =========================================================
# OPENAI LLM
# =========================================================

@st.cache_resource
def get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        return None

    return OpenAI(api_key=api_key)


def generate_ai_analysis(record):
    client = get_openai_client()

    if client is None:
        return (
            "LLM is not connected yet.\n\n"
            "Make sure OPENAI_API_KEY is set in Windows, then restart "
            "the terminal and Streamlit."
        )

    risk = record["Prediction"]

    prompt = f"""
You are an AI assistant inside a predictive maintenance application.

The machine prediction was generated by a CatBoost machine-learning model.
Do NOT change the model prediction. Explain it clearly using the provided
sensor values and calculated features.

Machine data:
- Machine Type: {record["Machine Type"]}
- Operating Mode: {record["Operating Mode"]}
- Operating Hour: {record["Operating Hour"]}
- Vibration RMS: {record["Vibration RMS"]}
- Motor Temperature: {record["Motor Temperature"]} °C
- Average Phase Current: {record["Average Phase Current"]} A
- Pressure Level: {record["Pressure Level"]}
- RPM: {record["RPM"]}
- Hours Since Maintenance: {record["Hours Since Maintenance"]}
- Ambient Temperature: {record["Ambient Temperature"]} °C
- Temperature Difference: {record["Temperature Difference"]}
- Vibration/RPM Ratio: {record["Vibration/RPM Ratio"]}
- Current/RPM Ratio: {record["Current/RPM Ratio"]}
- Pressure/RPM Ratio: {record["Pressure/RPM Ratio"]}
- Maintenance Load: {record["Maintenance Load"]}

CatBoost result:
- Prediction: {risk}
- Failure Probability: {record["Failure Probability"]}%
- Normal Probability: {record["Normal Probability"]}%

Write a concise professional explanation in English with exactly these sections:

Why this result?
Give 2-3 short points based only on the supplied values.

Recommended action
Give 2-3 practical maintenance actions. Do not claim a confirmed mechanical failure.

Important:
The CatBoost prediction is the actual prediction. The LLM only explains
the result and suggests actions.
"""

    try:
        response = client.responses.create(
            model="gpt-5.6-luna",
            instructions=(
                "You are a concise predictive-maintenance assistant. "
                "Never invent sensor readings or claim certainty."
            ),
            input=prompt
        )

        return response.output_text.strip()

    except Exception as e:
        return f"LLM error: {str(e)}"


st.markdown(
    '<div class="title">⚙️ Predictive Maintenance</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-powered machine failure prediction</div>',
    unsafe_allow_html=True
)


st.markdown(
    '<div class="card"><div class="section-title">🔧 Machine Information</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:
    machine_type = st.selectbox(
        "Machine Type",
        ["CNC", "Pump", "Compressor", "Robotic Arm"]
    )

with col2:
    operating_mode = st.selectbox(
        "Operating Mode",
        ["idle", "normal", "peak"]
    )

st.markdown("</div>", unsafe_allow_html=True)


st.markdown(
    '<div class="card"><div class="section-title">📊 Sensor Readings</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:
    vibration_rms = st.number_input(
        "Vibration RMS", min_value=0.0, value=1.0, step=0.01
    )

    temperature_motor = st.number_input(
        "Motor Temperature (°C)", value=70.0, step=0.1
    )

    current_phase_avg = st.number_input(
        "Average Phase Current (A)", min_value=0.0, value=10.0, step=0.1
    )

    pressure_level = st.number_input(
        "Pressure Level", min_value=0.0, value=5.0, step=0.1
    )

with col2:
    rpm = st.number_input(
        "RPM", min_value=0.0, value=1500.0, step=10.0
    )

    hours_since_maintenance = st.number_input(
        "Hours Since Maintenance", min_value=0.0, value=100.0, step=1.0
    )

    ambient_temp = st.number_input(
        "Ambient Temperature (°C)", value=25.0, step=0.1
    )

st.markdown("</div>", unsafe_allow_html=True)


st.markdown(
    '<div class="card"><div class="section-title">🕐 Time Information</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:
    timestamp = st.date_input("Date")

with col2:
    hour = st.selectbox("Operating Hour", list(range(24)), index=12)

st.markdown("</div>", unsafe_allow_html=True)


day_of_week = timestamp.weekday()
month = timestamp.month

temperature_difference = temperature_motor - ambient_temp
vibration_rpm_ratio = vibration_rms / (rpm + 1)
current_rpm_ratio = current_phase_avg / (rpm + 1)
pressure_rpm_ratio = pressure_level / (rpm + 1)
maintenance_load = hours_since_maintenance * rpm


if st.button("🔍 Predict Machine Failure"):

    input_data = pd.DataFrame({
        "machine_type": [machine_type],
        "vibration_rms": [vibration_rms],
        "temperature_motor": [temperature_motor],
        "current_phase_avg": [current_phase_avg],
        "pressure_level": [pressure_level],
        "rpm": [rpm],
        "operating_mode": [operating_mode],
        "hours_since_maintenance": [hours_since_maintenance],
        "ambient_temp": [ambient_temp],
        "hour": [hour],
        "day_of_week": [day_of_week],
        "month": [month],
        "temperature_difference": [temperature_difference],
        "vibration_rpm_ratio": [vibration_rpm_ratio],
        "current_rpm_ratio": [current_rpm_ratio],
        "pressure_rpm_ratio": [pressure_rpm_ratio],
        "maintenance_load": [maintenance_load]
    })

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    failure_percentage = probability * 100
    normal_percentage = 100 - failure_percentage

    # Prepare ALL entered inputs + prediction
    prediction_record = {
        "Saved At": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Machine Type": machine_type,
        "Operating Mode": operating_mode,
        "Date": str(timestamp),
        "Operating Hour": hour,
        "Vibration RMS": vibration_rms,
        "Motor Temperature": temperature_motor,
        "Average Phase Current": current_phase_avg,
        "Pressure Level": pressure_level,
        "RPM": rpm,
        "Hours Since Maintenance": hours_since_maintenance,
        "Ambient Temperature": ambient_temp,
        "Day of Week": day_of_week,
        "Month": month,
        "Temperature Difference": temperature_difference,
        "Vibration/RPM Ratio": vibration_rpm_ratio,
        "Current/RPM Ratio": current_rpm_ratio,
        "Pressure/RPM Ratio": pressure_rpm_ratio,
        "Maintenance Load": maintenance_load,
        "Prediction": "HIGH RISK" if prediction == 1 else "LOW RISK",
        "Failure Probability": round(failure_percentage, 2),
        "Normal Probability": round(normal_percentage, 2)
    }

    # Generate LLM explanation and recommendation
    with st.spinner("🤖 AI is analyzing the prediction..."):
        ai_analysis = generate_ai_analysis(prediction_record)

    prediction_record["AI Analysis"] = ai_analysis

    # Save everything, including the LLM explanation
    save_prediction(prediction_record)

    st.markdown(
        '<div class="card"><div class="section-title">🎯 Prediction Result</div>',
        unsafe_allow_html=True
    )

    if prediction == 1:
        st.markdown(
            f"""
            <div class="high-risk">
                <div style="font-size:50px;">🚨</div>
                <h1>HIGH RISK</h1>
                <h2>{failure_percentage:.2f}%</h2>
                <p>Machine failure is predicted within the next <strong>24 hours</strong>.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div class="low-risk">
                <div style="font-size:50px;">✅</div>
                <h1>LOW RISK</h1>
                <h2>{failure_percentage:.2f}%</h2>
                <p>No machine failure is predicted within the next <strong>24 hours</strong>.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="ai-card">
            <div class="ai-title">🤖 AI Analysis</div>
            <div class="ai-text">{ai_analysis}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Failure Probability", f"{failure_percentage:.2f}%")

    with col2:
        st.metric("Normal Probability", f"{normal_percentage:.2f}%")

    with st.expander("📋 View Input Data"):
        st.dataframe(
            input_data,
            use_container_width=True,
            hide_index=True
        )


# =========================================================
# PREDICTION HISTORY
# =========================================================

history = load_history()

if not history.empty:

    st.markdown(
        '<div class="card">'
        '<div class="history-title">📚 Prediction History</div>',
        unsafe_allow_html=True
    )

    st.caption("All previous predictions and their input values are saved automatically.")

    display_history = history.iloc[::-1].reset_index(drop=True)

    st.dataframe(
        display_history,
        use_container_width=True,
        hide_index=True
    )

    csv_data = history.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇️ Download History as CSV",
        data=csv_data,
        file_name="prediction_history.csv",
        mime="text/csv"
    )

    st.markdown("</div>", unsafe_allow_html=True)


st.markdown(
    """
    <div style="
        text-align:center;
        color:#82909c;
        font-size:13px;
        padding:30px 0 10px 0;
    ">
        Predictive Maintenance • CatBoost • 24-Hour Prediction
    </div>
    """,
    unsafe_allow_html=True
)

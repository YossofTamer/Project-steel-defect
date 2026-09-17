
import streamlit as st
import pandas as pd
from pathlib import Path
import os
import joblib
from datetime import datetime
try:
    from google import genai
except ImportError:
    genai = None

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =========================================================
# GEMINI CHAT STATE
# =========================================================
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if "current_prediction" not in st.session_state:
    st.session_state["current_prediction"] = None


# =========================================================
# PROFESSIONAL LIGHT UI
# =========================================================
st.markdown("""
<style>
    /* Global */
    .stApp {
        background: #f5f8fc;
    }

    .main .block-container {
        max-width: 1380px;
        padding: 2rem 2.2rem 3rem 2.2rem;
    }

    [data-testid="stHeader"] {
        background: rgba(245,248,252,0.85);
    }

    /* Typography */
    h1, h2, h3, p, label, div, span {
        font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    /* Header */
    .hero {
        background: linear-gradient(135deg, #ffffff 0%, #f1f7ff 100%);
        border: 1px solid #dce8f5;
        border-radius: 24px;
        padding: 28px 32px;
        margin-bottom: 22px;
        box-shadow: 0 8px 30px rgba(35, 72, 110, 0.07);
    }

    .hero-title {
        color: #142b45;
        font-size: 34px;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.8px;
    }

    .hero-subtitle {
        color: #64748b;
        font-size: 15px;
        margin-top: 7px;
    }

    .status-pill {
        display: inline-block;
        margin-top: 15px;
        padding: 7px 13px;
        border-radius: 999px;
        background: #eaf6f0;
        color: #16724b;
        border: 1px solid #ccebdc;
        font-size: 12px;
        font-weight: 700;
    }

    /* Section */
    .section-header {
        display: flex;
        align-items: center;
        gap: 10px;
        color: #17324d;
        font-size: 20px;
        font-weight: 800;
        margin: 12px 0 13px 2px;
    }

    .section-description {
        color: #718096;
        font-size: 13px;
        margin: -5px 0 15px 2px;
    }

    /* Cards */
    .card {
        background: #ffffff;
        border: 1px solid #e2eaf3;
        border-radius: 18px;
        padding: 21px 23px;
        margin-bottom: 18px;
        box-shadow: 0 5px 22px rgba(30, 65, 100, 0.055);
    }

    .mini-card {
        background: #ffffff;
        border: 1px solid #e2eaf3;
        border-radius: 16px;
        padding: 17px 19px;
        min-height: 105px;
        box-shadow: 0 4px 18px rgba(30, 65, 100, 0.045);
    }

    .mini-label {
        color: #718096;
        font-size: 12px;
        font-weight: 650;
        margin-bottom: 7px;
    }

    .mini-value {
        color: #142b45;
        font-size: 24px;
        font-weight: 800;
    }

    .mini-unit {
        color: #8a98a8;
        font-size: 12px;
        margin-left: 3px;
        font-weight: 500;
    }

    /* Inputs */
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {
        background: #ffffff !important;
        border: 1px solid #d7e2ee !important;
        border-radius: 11px !important;
        color: #172b3f !important;
        min-height: 44px;
    }

    input, textarea {
        color: #172b3f !important;
        background: #ffffff !important;
    }

    label {
        color: #334155 !important;
        font-size: 13px !important;
        font-weight: 650 !important;
    }

    /* Main button */
    .stButton > button {
        width: 100%;
        min-height: 50px;
        border-radius: 12px;
        border: 1px solid #0f766e;
        background: #0f766e;
        color: white;
        font-size: 15px;
        font-weight: 800;
        box-shadow: 0 7px 18px rgba(15,118,110,0.18);
        transition: 0.2s ease;
    }

    .stButton > button:hover {
        background: #0d665f;
        border-color: #0d665f;
        transform: translateY(-1px);
    }

    /* Risk cards */
    .risk-high {
        background: linear-gradient(135deg, #fff5f5, #fffafa);
        border: 1px solid #fecaca;
        border-left: 6px solid #dc2626;
        border-radius: 18px;
        padding: 23px 25px;
    }

    .risk-low {
        background: linear-gradient(135deg, #f0fdf8, #f9fffc);
        border: 1px solid #bbebd4;
        border-left: 6px solid #159a67;
        border-radius: 18px;
        padding: 23px 25px;
    }

    .risk-title {
        font-size: 26px;
        font-weight: 850;
        margin: 0;
    }

    .risk-high .risk-title { color: #b91c1c; }
    .risk-low .risk-title { color: #13734f; }

    .risk-number {
        color: #142b45;
        font-size: 42px;
        line-height: 1.05;
        font-weight: 850;
        margin: 8px 0;
    }

    .risk-text {
        color: #64748b;
        font-size: 13px;
        margin: 0;
    }

    /* Probability */
    .prob-wrap {
        background: #eef3f8;
        border-radius: 999px;
        height: 12px;
        overflow: hidden;
        margin: 9px 0 18px 0;
    }

    .prob-fill {
        height: 100%;
        border-radius: 999px;
    }

    .prob-label {
        display: flex;
        justify-content: space-between;
        color: #526477;
        font-size: 12px;
        font-weight: 650;
    }

    /* Tables */
    .history-note {
        color: #718096;
        font-size: 13px;
        margin-bottom: 12px;
    }

    /* Remove excess Streamlit top spacing */
    .element-container {
        margin-bottom: 0.35rem;
    }

    /* Chat readability */
    [data-testid="stChatMessage"] {
        border-radius: 14px;
        margin-bottom: 10px;
    }

    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
        color: #1f2937 !important;
    }

    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] li,
    [data-testid="stChatMessage"] span {
        color: #1f2937 !important;
        opacity: 1 !important;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #94a3b8;
        font-size: 12px;
        padding: 28px 0 5px 0;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# MODEL
# =========================================================
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

try:
    model = load_model()
except Exception as e:
    st.error("Could not load model.pkl")
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
# HEADER
# =========================================================
st.markdown("""
<div class="hero">
    <div class="hero-title">⚙️ Predictive Maintenance</div>
    <div class="hero-subtitle">
        Machine failure prediction dashboard powered by CatBoost
    </div>
    <div class="status-pill">● MODEL ONLINE &nbsp; • &nbsp; 24-HOUR FAILURE PREDICTION</div>
</div>
""", unsafe_allow_html=True)

# =========================================================
# MACHINE INFORMATION
# =========================================================
st.markdown('<div class="section-header">🔧 Machine Configuration</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-description">Select the machine and its current operating condition.</div>',
    unsafe_allow_html=True
)

st.markdown('<div class="card">', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)

with c1:
    machine_type = st.selectbox(
        "Machine Type",
        ["CNC", "Pump", "Compressor", "Robotic Arm"]
    )

with c2:
    operating_mode = st.selectbox(
        "Operating Mode",
        ["idle", "normal", "peak"]
    )

with c3:
    hour = st.selectbox(
        "Operating Hour",
        list(range(24)),
        index=12,
        format_func=lambda x: f"{x:02d}:00"
    )

st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# SENSOR READINGS
# =========================================================
st.markdown('<div class="section-header">📡 Live Sensor Readings</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-description">Enter the latest measurements collected from the machine.</div>',
    unsafe_allow_html=True
)

st.markdown('<div class="card">', unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)

with c1:
    vibration_rms = st.number_input(
        "Vibration RMS", min_value=0.0, value=1.0, step=0.01
    )

with c2:
    temperature_motor = st.number_input(
        "Motor Temperature (°C)", value=70.0, step=0.1
    )

with c3:
    current_phase_avg = st.number_input(
        "Average Phase Current (A)", min_value=0.0, value=10.0, step=0.1
    )

with c4:
    pressure_level = st.number_input(
        "Pressure Level", min_value=0.0, value=5.0, step=0.1
    )

c5, c6, c7, c8 = st.columns(4)

with c5:
    rpm = st.number_input(
        "RPM", min_value=0.0, value=1500.0, step=10.0
    )

with c6:
    hours_since_maintenance = st.number_input(
        "Hours Since Maintenance", min_value=0.0, value=100.0, step=1.0
    )

with c7:
    ambient_temp = st.number_input(
        "Ambient Temperature (°C)", value=25.0, step=0.1
    )

with c8:
    timestamp = st.date_input("Date", value=datetime.now().date())

st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# LIVE FEATURE PREVIEW
# =========================================================
temperature_difference = temperature_motor - ambient_temp
vibration_rpm_ratio = vibration_rms / (rpm + 1)
current_rpm_ratio = current_phase_avg / (rpm + 1)
pressure_rpm_ratio = pressure_level / (rpm + 1)
maintenance_load = hours_since_maintenance * rpm
day_of_week = timestamp.weekday()
month = timestamp.month

st.markdown('<div class="section-header">🧮 Calculated Features</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-description">Features generated automatically before the model prediction.</div>',
    unsafe_allow_html=True
)

feature_cols = st.columns(5)

features = [
    ("Temperature Δ", f"{temperature_difference:.2f}", "°C"),
    ("Vibration / RPM", f"{vibration_rpm_ratio:.5f}", ""),
    ("Current / RPM", f"{current_rpm_ratio:.5f}", ""),
    ("Pressure / RPM", f"{pressure_rpm_ratio:.5f}", ""),
    ("Maintenance Load", f"{maintenance_load:,.0f}", ""),
]

for col, (label, value, unit) in zip(feature_cols, features):
    with col:
        st.markdown(
            f"""
            <div class="mini-card">
                <div class="mini-label">{label}</div>
                <div class="mini-value">{value}<span class="mini-unit">{unit}</span></div>
            </div>
            """,
            unsafe_allow_html=True
        )

st.write("")

# =========================================================
# PREDICT
# =========================================================
predict_col1, predict_col2, predict_col3 = st.columns([1, 2, 1])
with predict_col2:
    predict_clicked = st.button("🔍  Predict Machine Failure", use_container_width=True)


# =========================================================
# GEMINI CONFIGURATION
# =========================================================


GEMINI_MODEL = "gemini-3.6-flash"
GEMINI_FALLBACK_MODEL = "gemini-3.5-flash-lite"

GEMINI_SYSTEM_PROMPT = """
You are a professional Predictive Maintenance AI Assistant.
You analyze the current machine data and explain the CatBoost prediction.

Rules:
- Use only the machine data and prediction supplied by the application.
- Never invent sensor readings, probabilities, history, or failures.
- Do not change or override the CatBoost prediction.
- A prediction is not proof that a mechanical failure has occurred.
- Give practical, concise maintenance guidance.
- Explain technical concepts simply when appropriate.
"""

def get_gemini_client():
    if genai is None:
        return None

    # Try st.secrets first (Streamlit Cloud), then fall back to env var (local)
    api_key = ""
    try:
        api_key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        pass
    if not api_key:
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        return None

    return genai.Client(api_key=api_key)

def get_recent_history():
    history = load_history()
    if history.empty:
        return "No previous prediction history is available."
    return str(history.tail(10).to_dict(orient="records"))

def generate_gemini_response(client, prompt):
    """Call Gemini with automatic retry and a stable fallback model.

    503/UNAVAILABLE can happen when a model is temporarily under heavy load.
    Retry briefly, then fall back to Gemini 3.5 Flash-Lite.
    """
    import time

    last_error = None
    for attempt in range(3):
        try:
            return client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config={"system_instruction": GEMINI_SYSTEM_PROMPT}
            )
        except Exception as e:
            last_error = e
            error_text = str(e)
            is_temporary = (
                "503" in error_text
                or "UNAVAILABLE" in error_text
                or "high demand" in error_text.lower()
                or "temporarily" in error_text.lower()
            )
            if not is_temporary:
                raise
            if attempt < 2:
                time.sleep(2 ** attempt)

    # Fallback model after repeated 503 errors.
    try:
        return client.models.generate_content(
            model=GEMINI_FALLBACK_MODEL,
            contents=prompt,
            config={"system_instruction": GEMINI_SYSTEM_PROMPT}
        )
    except Exception:
        raise last_error

def machine_context(record):
    if not record:
        return "No current prediction has been generated yet."

    return f"""
Machine Type: {record.get("Machine Type")}
Operating Mode: {record.get("Operating Mode")}
Date: {record.get("Date")}
Operating Hour: {record.get("Operating Hour")}
Vibration RMS: {record.get("Vibration RMS")}
Motor Temperature: {record.get("Motor Temperature")} °C
Average Phase Current: {record.get("Average Phase Current")} A
Pressure Level: {record.get("Pressure Level")}
RPM: {record.get("RPM")}
Hours Since Maintenance: {record.get("Hours Since Maintenance")}
Ambient Temperature: {record.get("Ambient Temperature")} °C
Temperature Difference: {record.get("Temperature Difference")}
Vibration/RPM Ratio: {record.get("Vibration/RPM Ratio")}
Current/RPM Ratio: {record.get("Current/RPM Ratio")}
Pressure/RPM Ratio: {record.get("Pressure/RPM Ratio")}
Maintenance Load: {record.get("Maintenance Load")}
Prediction: {record.get("Prediction")}
Failure Probability: {record.get("Failure Probability")}%
Normal Probability: {record.get("Normal Probability")}%
"""

def ask_gemini(question, current_record=None):
    client = get_gemini_client()

    if client is None:
        return (
            "⚠️ Gemini is not connected.\n\n"
            "Run `pip install -U google-genai` and set `GEMINI_API_KEY` "
            "before starting Streamlit."
        )

    prompt = f"""
CURRENT MACHINE DATA:
{machine_context(current_record)}

RECENT PREDICTION HISTORY:
{get_recent_history()}

USER QUESTION:
{question}

Answer directly and professionally. Use the current machine data when relevant.
"""
    try:
        response = generate_gemini_response(client, prompt)
        return response.text.strip()
    except Exception as e:
        error_text = str(e)
        if "503" in error_text or "UNAVAILABLE" in error_text or "high demand" in error_text.lower():
            return (
                "⚠️ Gemini is temporarily busy (503). The app already retried "
                "and tried the fallback Gemini 3.5 Flash-Lite model. Please try "
                "Predict again in a few seconds."
            )
        if "404" in error_text and ("gemini-2.5-flash" in error_text or "not found" in error_text.lower()):
            return (
                "⚠️ Gemini model was unavailable. The app is configured for "
                "Gemini 3.6 Flash with a fallback model. Restart Streamlit and try again."
            )
        return f"⚠️ Gemini error: {error_text}"

def gemini_analysis(record):
    """Generate the AI Analysis for the current prediction using Gemini."""
    client = get_gemini_client()

    if client is None:
        return (
            "⚠️ Gemini is not connected.\n\n"
            "Set the GEMINI_API_KEY environment variable and restart Streamlit."
        )

    prompt = f"""
CURRENT MACHINE DATA
{machine_context(record)}

RECENT PREDICTION HISTORY
{get_recent_history()}

Analyze this CatBoost prediction for the user.

Use exactly these sections:

### Why this result?
Give 2-4 concise points based ONLY on the supplied machine data.

### Recommended action
Give 2-4 practical predictive-maintenance actions.

### Important note
State briefly that the CatBoost result is a prediction and does not confirm
that a mechanical failure has occurred.

Do not invent any values or facts.
"""

    try:
        response = generate_gemini_response(client, prompt)
        return response.text.strip()
    except Exception as e:
        error_text = str(e)
        if "503" in error_text or "UNAVAILABLE" in error_text or "high demand" in error_text.lower():
            return (
                "⚠️ Gemini is temporarily busy (503). The app already retried "
                "and tried the fallback Gemini 3.5 Flash-Lite model. Please try "
                "Predict again in a few seconds."
            )
        if "404" in error_text and ("gemini-2.5-flash" in error_text or "not found" in error_text.lower()):
            return (
                "⚠️ Gemini model was unavailable. The app is configured for "
                "Gemini 3.6 Flash with a fallback model. Restart Streamlit and try again."
            )
        return f"⚠️ Gemini error: {error_text}"


# =========================================================
# PREDICTION
# =========================================================
if predict_clicked:

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

    try:
        with st.spinner("Running predictive model..."):
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0][1]
    except Exception as e:
        st.error("Prediction failed.")
        st.code(str(e))
        st.stop()

    failure_percentage = probability * 100
    normal_percentage = 100 - failure_percentage

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

    save_prediction(prediction_record)
    st.session_state["current_prediction"] = prediction_record

    # =====================================================
    # RESULT
    # =====================================================
    st.markdown("---")
    st.markdown('<div class="section-header">🎯 Prediction Result</div>', unsafe_allow_html=True)

    if prediction == 1:
        st.markdown(
            f"""
            <div class="risk-high">
                <div class="risk-title">🚨 HIGH RISK</div>
                <div class="risk-number">{failure_percentage:.2f}%</div>
                <p class="risk-text">
                    The model predicts a machine failure within the next 24 hours.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div class="risk-low">
                <div class="risk-title">✅ LOW RISK</div>
                <div class="risk-number">{failure_percentage:.2f}%</div>
                <p class="risk-text">
                    The model does not predict a machine failure within the next 24 hours.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.write("")

    # Probability cards
    p1, p2 = st.columns(2)

    with p1:
        st.markdown(
            f"""
            <div class="card">
                <div class="mini-label">FAILURE PROBABILITY</div>
                <div class="mini-value">{failure_percentage:.2f}%</div>
                <div class="prob-wrap">
                    <div class="prob-fill" style="width:{min(failure_percentage,100):.2f}%; background:#dc2626;"></div>
                </div>
                <div class="prob-label">
                    <span>Predicted failure</span>
                    <span>{failure_percentage:.2f}%</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with p2:
        st.markdown(
            f"""
            <div class="card">
                <div class="mini-label">NORMAL PROBABILITY</div>
                <div class="mini-value">{normal_percentage:.2f}%</div>
                <div class="prob-wrap">
                    <div class="prob-fill" style="width:{min(normal_percentage,100):.2f}%; background:#159a67;"></div>
                </div>
                <div class="prob-label">
                    <span>Normal operation</span>
                    <span>{normal_percentage:.2f}%</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # =====================================================
    # GEMINI AI ANALYSIS
    # =====================================================
    st.markdown('<div class="section-header">🤖 Gemini AI Analysis</div>', unsafe_allow_html=True)

    with st.spinner("Gemini is analyzing the prediction..."):
        ai_analysis = gemini_analysis(prediction_record)

    st.markdown(
        f"""
        <div class="card">
            <div style="color:#2563eb;font-size:16px;font-weight:800;margin-bottom:10px;">
                Gemini LLM Interpretation
            </div>
            <div style="color:#334155;font-size:14px;line-height:1.75;white-space:pre-wrap;">
                {ai_analysis}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Sensor snapshot
    st.markdown('<div class="section-header">📊 Sensor Snapshot</div>', unsafe_allow_html=True)

    s1, s2, s3, s4, s5 = st.columns(5)
    snapshot = [
        ("Vibration", f"{vibration_rms:.2f}", "RMS"),
        ("Temperature", f"{temperature_motor:.1f}", "°C"),
        ("Current", f"{current_phase_avg:.1f}", "A"),
        ("Pressure", f"{pressure_level:.2f}", ""),
        ("RPM", f"{rpm:,.0f}", "RPM")
    ]

    for col, (label, value, unit) in zip([s1, s2, s3, s4, s5], snapshot):
        with col:
            st.markdown(
                f"""
                <div class="mini-card">
                    <div class="mini-label">{label}</div>
                    <div class="mini-value">{value}<span class="mini-unit">{unit}</span></div>
                </div>
                """,
                unsafe_allow_html=True
            )

    # Input details
    st.write("")
    with st.expander("📋 View Model Input & Engineered Features"):
        st.dataframe(
            input_data,
            use_container_width=True,
            hide_index=True
        )

# =========================================================
# GEMINI AI CHATBOT
# =========================================================
st.markdown("---")
st.markdown('<div class="section-header">💬 Gemini AI Maintenance Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-description">Talk naturally with Gemini about the current machine, prediction, sensors, and maintenance.</div>',
    unsafe_allow_html=True
)

with st.container():
    quick_cols = st.columns(4)
    quick_questions = [
        "Why is this machine high risk?",
        "Which sensor is most concerning?",
        "What maintenance should I do?",
        "Explain the prediction simply."
    ]

    for i, q in enumerate(quick_questions):
        if quick_cols[i].button(q, key=f"gemini_quick_{i}"):
            st.session_state["chat_history"].append({"role": "user", "content": q})
            answer = ask_gemini(q, st.session_state.get("current_prediction"))
            st.session_state["chat_history"].append({"role": "assistant", "content": answer})
            st.rerun()

    for message in st.session_state.get("chat_history", []):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

if st.session_state.get("chat_history"):
    if st.button("🗑️ Clear Conversation", key="clear_gemini_chat"):
        st.session_state["chat_history"] = []
        st.rerun()

question = st.chat_input("Ask Gemini about your machine...")
if question:
    st.session_state["chat_history"].append({"role": "user", "content": question})
    answer = ask_gemini(question, st.session_state.get("current_prediction"))
    st.session_state["chat_history"].append({"role": "assistant", "content": answer})
    st.rerun()

# =========================================================
# HISTORY
# =========================================================
st.markdown("---")
st.markdown('<div class="section-header">📚 Prediction History</div>', unsafe_allow_html=True)

history = load_history()

if history.empty:
    st.info("No predictions have been saved yet. Run your first prediction above.")
else:
    st.markdown(
        f'<div class="history-note">{len(history)} prediction(s) saved automatically.</div>',
        unsafe_allow_html=True
    )

    display_history = history.iloc[::-1].reset_index(drop=True)

    # Compact summary
    h1, h2, h3 = st.columns(3)

    total = len(history)
    high_count = int((history["Prediction"] == "HIGH RISK").sum()) if "Prediction" in history.columns else 0
    low_count = total - high_count

    with h1:
        st.markdown(
            f"""
            <div class="mini-card">
                <div class="mini-label">TOTAL PREDICTIONS</div>
                <div class="mini-value">{total}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with h2:
        st.markdown(
            f"""
            <div class="mini-card">
                <div class="mini-label">HIGH RISK</div>
                <div class="mini-value">{high_count}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with h3:
        st.markdown(
            f"""
            <div class="mini-card">
                <div class="mini-label">LOW RISK</div>
                <div class="mini-value">{low_count}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.write("")

    st.dataframe(
        display_history,
        use_container_width=True,
        hide_index=True
    )

    csv_data = history.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇️ Download Prediction History (CSV)",
        data=csv_data,
        file_name="prediction_history.csv",
        mime="text/csv",
        use_container_width=False
    )

# =========================================================
# FOOTER
# =========================================================
st.markdown("""
<div class="footer">
    Predictive Maintenance Dashboard &nbsp;•&nbsp; CatBoost &nbsp;•&nbsp; 24-Hour Failure Prediction
</div>
""", unsafe_allow_html=True)

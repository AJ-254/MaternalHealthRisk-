import streamlit as st
import joblib
import numpy as np

# === Load model and scaler ===
model = joblib.load("maternal_risk_model.pkl")
scaler = joblib.load("scaler.pkl")

# === App Title ===
st.set_page_config(page_title="Maternal Health Risk Predictor", page_icon="👩‍🍼")
st.title("👩‍🍼 Maternal Health Risk Prediction App")
st.markdown("Use this AI-powered tool to estimate maternal health risk based on key medical indicators.")

# === Sidebar Info ===
st.sidebar.header("About the App")
st.sidebar.info(
    "This predictive model uses health indicators such as blood pressure, blood sugar, and heart rate "
    "to classify maternal health risk into **Low**, **Medium**, or **High** categories.\n\n"
    "⚠️ This AI tool is for research and educational purposes only and does not replace professional medical judgment.."
)

# === Input Fields ===
st.subheader("Enter Patient Health Data")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=12, max_value=60)
    body_temp = st.number_input("Body Temperature (°C)", 35.0, 40.0, 36.8)

with col2:
    systolic_bp = st.number_input("Systolic BP (mmHg)", 90, 200, 120)
    diastolic_bp = st.number_input("Diastolic BP (mmHg)", 60, 120, 80)

with col3:
    bs = st.number_input("Blood Sugar (mmol/L)", 2.0, 15.0, 4.5)
    heart_rate = st.number_input("Heart Rate (bpm)", 50, 150, 75)

# === Prediction Button ===
if st.button("🔍 Predict Risk Level"):

    # Fill optional fields with normal clinical defaults
    if bs is None:
        bs = 4.5   # Normal fasting glucose

    if body_temp is None:
        body_temp = 36.8  # Normal body temperature

    # Prepare input
    input_data = np.array([[age, systolic_bp, diastolic_bp, bs, body_temp, heart_rate]])
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)[0]
    risk_labels = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}
    risk_level = risk_labels[prediction]

    # Display Result
    st.markdown("---")

    if risk_level == "Low Risk":
        st.success(
            "✅ **Low Risk Identified**\n\n"
            "Current indicators fall within low-risk thresholds. Continue routine antenatal care and healthy lifestyle practices."
        )

    elif risk_level == "Medium Risk":
        st.warning(
            "⚠️ **Moderate Risk Identified**\n\n"
            "Clinical parameters suggest moderate risk. Closer monitoring and regular antenatal follow-ups are recommended."
        )

    else:
        st.error(
            "🚨 **High Risk Identified**\n\n"
            "Indicators suggest elevated maternal risk. Immediate clinical assessment and intervention are strongly advised."
        )

    st.markdown("---")

st.info("This AI tool is for research and educational purposes only and does not replace professional medical judgment.")



# === Footer ===



st.caption("AI-powered risk assessment tool for antenatal screening.")

st.caption("Developed by **Juliet Asiedu - RM🩺**💡")

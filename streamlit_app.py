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
    "⚠️ This tool is for educational purposes only and should not replace professional medical advice."
)

# === Input Fields ===
st.subheader("Enter Patient Health Data")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age (years)", 15, 50, 25)
    body_temp = st.number_input("Body Temperature (°C)", 35.0, 40.0, 36.8)

with col2:
    systolic_bp = st.number_input("Systolic BP (mmHg)", 90, 200, 120)
    diastolic_bp = st.number_input("Diastolic BP (mmHg)", 60, 120, 80)

with col3:
    bs = st.number_input("Blood Sugar (mmol/L)", 2.0, 15.0, 4.5)
    heart_rate = st.number_input("Heart Rate (bpm)", 50, 150, 75)

# === Prediction Button ===
if st.button("🔍 Predict Risk Level"):
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
        st.success("✅ Prediction: **Low Risk**\n\nEverything looks good! Keep maintaining a healthy lifestyle.")
    elif risk_level == "Medium Risk":
        st.warning("⚠️ Prediction: **Medium Risk**\n\nRegular checkups and monitoring are advised.")
    else:
        st.error("🚨 Prediction: **High Risk**\n\nSeek immediate medical attention for proper evaluation.")

    st.markdown("---")

# === Footer ===
st.caption("Developed as part of the **AI for Software Engineering** course | Week 5 Assignment 💡")

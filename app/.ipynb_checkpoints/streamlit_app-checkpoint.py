# streamlit_app.py
import streamlit as st
import pandas as pd
import joblib
import os

# 🔹 Load model using relative path
model_path = os.path.join(os.path.dirname(__file__), "health_risk_rf_model.pkl")

try:
    model = joblib.load(model_path)
except FileNotFoundError:
    st.error("❌ Model file not found! Please make sure 'health_risk_rf_model.pkl' is in the app/ folder.")
    st.stop()

# 🔹 App title
st.title("🩺 Health Risk Predictor")

# 🔹 Sidebar user inputs
st.sidebar.header("Enter your health data:")
age = st.sidebar.number_input("Age", min_value=0, max_value=120, value=30)
weight = st.sidebar.number_input("Weight (kg)", min_value=1, max_value=300, value=70)
bp = st.sidebar.number_input("Blood Pressure", min_value=50, max_value=250, value=120)
cholesterol = st.sidebar.number_input("Cholesterol", min_value=50, max_value=400, value=180)
glucose = st.sidebar.number_input("Glucose", min_value=50, max_value=400, value=100)

# 🔹 Create input dataframe
input_data = pd.DataFrame({
    "age": [age],
    "weight": [weight],
    "bp": [bp],
    "cholesterol": [cholesterol],
    "glucose": [glucose]
})

# 🔹 Predict
prediction = model.predict(input_data)[0]

# 🔹 Display result
st.subheader("Prediction:")
if prediction == 1:
    st.warning("⚠️ High health risk")
else:
    st.success("✅ Low health risk")

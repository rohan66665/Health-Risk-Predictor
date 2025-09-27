import streamlit as st
import joblib
import pandas as pd

# Load model
model = joblib.load("health_risk_rf_model.pkl")

st.title("Health Risk Predictor")

# User input
age = st.number_input("Age", 0, 120, 30)
weight = st.number_input("Weight", 0, 200, 70)
bp = st.number_input("Blood Pressure", 50, 200, 120)
chol = st.number_input("Cholesterol", 100, 300, 180)
glucose = st.number_input("Glucose", 50, 300, 100)

if st.button("Predict Risk"):
    input_data = pd.DataFrame([[age, weight, bp, chol, glucose]], 
                              columns=["age", "weight", "bp", "cholesterol", "glucose"])
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Health Risk: {'High' if prediction==1 else 'Low'}")
import streamlit as st
import pandas as pd
import joblib

# 1️⃣ Page config
st.set_page_config(page_title="Health Risk Predictor", page_icon="🩺", layout="centered")

# 2️⃣ Load model
@st.cache_resource
def load_model():
    return joblib.load("health_risk_rf_model.pkl")

model = load_model()

st.title("🩺 Health Risk Prediction App")
st.write("Fill in the details below to check **Health Risk** using our trained Random Forest model.")

# 3️⃣ Input form
with st.form("health_form"):
    age = st.number_input("Age", min_value=10, max_value=100, value=30)
    weight = st.number_input("Weight (kg)", min_value=20, max_value=200, value=70)
    bp = st.number_input("Blood Pressure", min_value=60, max_value=200, value=120)
    cholesterol = st.number_input("Cholesterol Level", min_value=100, max_value=400, value=200)
    glucose = st.number_input("Glucose Level", min_value=50, max_value=250, value=100)

    submitted = st.form_submit_button("Predict Risk")

# 4️⃣ Prediction
if submitted:
    input_data = pd.DataFrame([[age, weight, bp, cholesterol, glucose]],
                              columns=["age", "weight", "bp", "cholesterol", "glucose"])
    
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.error("⚠️ High Risk of Health Issues Detected!")
    else:
        st.success("✅ Low Risk — You seem healthy!")

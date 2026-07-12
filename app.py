from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path("models/diabetes_model.joblib")

st.set_page_config(page_title="Diabetes Risk Predictor")
st.title("Diabetes Risk Prediction Demo")
st.warning(
    "Educational use only. This application is not a diagnostic tool "
    "and must not be used for medical decisions."
)

if not MODEL_PATH.exists():
    st.error("Model not found. Run: python -m src.train")
    st.stop()

model = joblib.load(MODEL_PATH)

with st.form("prediction_form"):
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=25, value=0)
    glucose = st.number_input("Glucose", min_value=0.0, max_value=300.0, value=100.0)
    blood_pressure = st.number_input("Blood Pressure", min_value=0.0, max_value=200.0, value=70.0)
    skin_thickness = st.number_input("Skin Thickness", min_value=0.0, max_value=100.0, value=20.0)
    insulin = st.number_input("Insulin", min_value=0.0, max_value=1000.0, value=80.0)
    bmi = st.number_input("BMI", min_value=0.0, max_value=80.0, value=25.0)
    pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
    age = st.number_input("Age", min_value=1, max_value=120, value=30)
    submitted = st.form_submit_button("Estimate probability")

if submitted:
    row = pd.DataFrame([{
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": pedigree,
        "Age": age,
    }])
    probability = float(model.predict_proba(row)[0, 1])
    st.metric("Model-estimated probability", f"{probability:.1%}")
    st.caption("This is a model output, not a diagnosis.")

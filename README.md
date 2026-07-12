# Diabetes Risk Prediction System

A beginner-friendly machine learning project for learning Python, preprocessing,
classification, evaluation, and simple deployment.

> Educational use only. This is not a medical device and must not be used to diagnose diabetes.

## Expected dataset

Place `diabetes.csv` in the `data` folder with these columns:

`Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, Outcome`

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
.venv\Scripts\activate    # Windows
pip install -r requirements.txt
```

## Train

```bash
python -m src.train
```

## Run the app

```bash
streamlit run app.py
```

[README.md](https://github.com/user-attachments/files/29946055/README.md)
# Diabetes Risk Prediction System

A beginner-friendly machine learning project for learning Python, data
preprocessing, classification, model evaluation, and simple deployment with
Streamlit.

> ⚠️ **Educational use only.** This is not a medical device and must not be
> used to diagnose, treat, or screen for diabetes. Always consult a qualified
> healthcare professional for medical advice.

---

## Overview

This project walks through a complete, minimal ML workflow:

1. Load and clean the [Pima Indians Diabetes dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
2. Preprocess features (handle missing/zero values, scale inputs)
3. Train a classification model to predict diabetes risk
4. Evaluate performance with standard classification metrics
5. Serve predictions through a simple Streamlit web app

## Project structure

```
.
├── data/
│   └── diabetes.csv        # dataset (not included, see below)
├── src/
│   ├── train.py             # training entry point
│   ├── preprocess.py        # data cleaning / feature prep
│   └── evaluate.py          # metrics and reporting
├── models/                  # saved model artifacts (created after training)
├── app.py                   # Streamlit app
├── requirements.txt
└── README.md
```

*(Adjust this tree to match your actual repo layout.)*

## Dataset

Download the dataset and place it at `data/diabetes.csv`. It must contain
the following columns:

| Column | Description |
|---|---|
| `Pregnancies` | Number of times pregnant |
| `Glucose` | Plasma glucose concentration |
| `BloodPressure` | Diastolic blood pressure (mm Hg) |
| `SkinThickness` | Triceps skinfold thickness (mm) |
| `Insulin` | 2-Hour serum insulin (mu U/ml) |
| `BMI` | Body mass index |
| `DiabetesPedigreeFunction` | Diabetes likelihood based on family history |
| `Age` | Age in years |
| `Outcome` | Target label (1 = diabetic, 0 = non-diabetic) |

## Setup

Requires Python 3.9+.

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate      # macOS/Linux
.venv\Scripts\activate         # Windows

# 2. Install dependencies
pip install -r requirements.txt
```

## Usage

### Train the model

```bash
python -m src.train
```

This will preprocess the data, train the model, print evaluation metrics,
and save the trained model to `models/`.

### Run the app

```bash
streamlit run app.py
```

Opens a local web interface where you can enter patient values and get a
predicted risk score.

## Results

*(Fill in after training — e.g. accuracy, precision/recall, ROC-AUC, and a
short note on which model performed best.)*

| Metric | Score |
|---|---|
| Accuracy | — |
| Precision | — |
| Recall | — |
| ROC-AUC | — |

## Limitations

- Trained on a small, non-diverse historical dataset (Pima Indians Diabetes
  dataset), so predictions may not generalize to other populations.
- Not validated for clinical use.
- Missing/zero values in some fields (e.g. `Insulin`, `SkinThickness`) are
  imputed, which introduces some uncertainty.

## License

*(Add your license here, e.g. MIT.)*

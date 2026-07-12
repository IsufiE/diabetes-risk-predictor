from pathlib import Path

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


MODEL_PATH = Path("models/diabetes_model.joblib")

# Risk bands used for both the gauge and the result message.
# (Thresholds are illustrative — tune them to your model's calibration.)
RISK_BANDS = [
    (0.33, "Low", "#2ecc71"),
    (0.66, "Moderate", "#f39c12"),
    (1.01, "High", "#e74c3c"),
]


def get_risk_band(probability: float):
    for threshold, label, color in RISK_BANDS:
        if probability < threshold:
            return label, color
    return RISK_BANDS[-1][1], RISK_BANDS[-1][2]


st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="wide",
)

# --- Light custom styling -----------------------------------------------
st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; }
    div[data-testid="stMetricValue"] { font-size: 2.2rem; }
    .risk-card {
        padding: 1.25rem 1.5rem;
        border-radius: 0.75rem;
        border: 1px solid rgba(128,128,128,0.25);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    """Load the trained machine-learning pipeline once and reuse it."""
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


model = load_model()

st.title("🩺 Diabetes Risk Prediction Demo")
st.caption(
    "A machine-learning model estimates diabetes risk from a handful of "
    "health measurements."
)
st.warning(
    "This project is for educational purposes only. It is not a medical "
    "device and must not be used for diagnosis or treatment decisions.",
    icon="⚠️",
)

if model is None:
    st.error(
        "The trained model could not be found. "
        "Run `python3 -m src.train` before starting the application."
    )
    st.stop()

# --- Inputs live in the sidebar so the main area is reserved for results ---
with st.sidebar:
    st.header("Patient information")

    pregnancies = st.number_input(
        "Pregnancies", min_value=0, max_value=25, value=0,
        help="Number of previous pregnancies.",
    )
    glucose = st.slider(
        "Glucose", min_value=0.0, max_value=300.0, value=100.0,
        help="Blood glucose measurement.",
    )
    blood_pressure = st.slider(
        "Blood pressure", min_value=0.0, max_value=200.0, value=70.0,
        help="Diastolic blood pressure measurement.",
    )
    skin_thickness = st.slider(
        "Skin thickness", min_value=0.0, max_value=100.0, value=20.0,
        help="Skin-fold thickness measurement.",
    )
    insulin = st.slider(
        "Insulin", min_value=0.0, max_value=1000.0, value=80.0,
        help="Insulin measurement.",
    )
    bmi = st.slider(
        "BMI", min_value=0.0, max_value=80.0, value=25.0,
        help="Body Mass Index.",
    )
    pedigree = st.slider(
        "Diabetes pedigree function", min_value=0.0, max_value=3.0, value=0.5,
        help="A value representing family-history-related diabetes risk.",
    )
    age = st.number_input(
        "Age", min_value=1, max_value=120, value=30,
        help="Patient age in years.",
    )

    predict_button = st.button(
        "Estimate risk", type="primary", use_container_width=True,
    )

# --- Prediction + results ------------------------------------------------
if not predict_button:
    st.info("Enter patient information in the sidebar, then click **Estimate risk**.")
    st.stop()

input_data = pd.DataFrame(
    [
        {
            "Pregnancies": pregnancies,
            "Glucose": glucose,
            "BloodPressure": blood_pressure,
            "SkinThickness": skin_thickness,
            "Insulin": insulin,
            "BMI": bmi,
            "DiabetesPedigreeFunction": pedigree,
            "Age": age,
        }
    ]
)

probability = float(model.predict_proba(input_data)[0, 1])
risk_label, risk_color = get_risk_band(probability)

st.divider()
st.subheader("Model result")

gauge_col, detail_col = st.columns([1, 1.4])

with gauge_col:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": risk_color},
                "steps": [
                    {"range": [0, 33], "color": "#eafaf1"},
                    {"range": [33, 66], "color": "#fef5e7"},
                    {"range": [66, 100], "color": "#fdedec"},
                ],
            },
        )
    )
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)

with detail_col:
    st.markdown(
        f"""
        <div class="risk-card" style="border-left: 6px solid {risk_color};">
            <span style="font-size:0.9rem; color:gray;">Risk category</span><br>
            <span style="font-size:1.8rem; font-weight:600; color:{risk_color};">
                {risk_label}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")
    st.caption(
        "This reflects patterns learned from the training dataset. "
        "It is not a diagnosis and should not be interpreted as medical advice."
    )

    with st.expander("View submitted values"):
        st.dataframe(input_data, use_container_width=True, hide_index=True)

# --- Feature importance ---------------------------------------------------
st.divider()
st.subheader("What drove this result")

model_step = model.named_steps["model"]

if hasattr(model_step, "feature_importances_"):
    importance_df = pd.DataFrame(
        {
            "Feature": input_data.columns,
            "Importance": model_step.feature_importances_,
        }
    ).sort_values("Importance", ascending=True)

    bar_fig = go.Figure(
        go.Bar(
            x=importance_df["Importance"],
            y=importance_df["Feature"],
            orientation="h",
            marker=dict(
                color=importance_df["Importance"],
                colorscale="Blues",
            ),
        )
    )
    bar_fig.update_layout(
        height=350,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="Importance",
        yaxis_title=None,
    )
    st.plotly_chart(bar_fig, use_container_width=True)

    most_important_feature = importance_df.iloc[-1]["Feature"]
    st.info(f"The model's most influential feature is **{most_important_feature}**.")
    st.caption(
        "Feature importance shows which inputs were most useful to the model. "
        "It does not prove that a feature caused diabetes."
    )
else:
    st.info("Feature importance is not available for the selected model.")
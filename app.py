from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


MODEL_PATH = Path("models/diabetes_model.joblib")


st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="wide",
)


@st.cache_resource
def load_model():
    """Load the trained machine-learning pipeline once and reuse it."""
    if not MODEL_PATH.exists():
        return None

    return joblib.load(MODEL_PATH)


model = load_model()


st.title("🩺 Diabetes Risk Prediction Demo")

st.write(
    """
    This application uses a machine-learning model to estimate diabetes risk
    based on several health measurements.
    """
)

st.warning(
    """
    This project is for educational purposes only. It is not a medical device
    and must not be used for diagnosis or treatment decisions.
    """
)


if model is None:
    st.error(
        "The trained model could not be found. "
        "Run `python3 -m src.train` before starting the application."
    )
    st.stop()


st.subheader("Patient information")

left_column, right_column = st.columns(2)


with left_column:
    pregnancies = st.number_input(
        "Pregnancies",
        min_value=0,
        max_value=25,
        value=0,
        help="Number of previous pregnancies.",
    )

    glucose = st.number_input(
        "Glucose",
        min_value=0.0,
        max_value=300.0,
        value=100.0,
        help="Blood glucose measurement.",
    )

    blood_pressure = st.number_input(
        "Blood pressure",
        min_value=0.0,
        max_value=200.0,
        value=70.0,
        help="Diastolic blood pressure measurement.",
    )

    skin_thickness = st.number_input(
        "Skin thickness",
        min_value=0.0,
        max_value=100.0,
        value=20.0,
        help="Skin-fold thickness measurement.",
    )


with right_column:
    insulin = st.number_input(
        "Insulin",
        min_value=0.0,
        max_value=1000.0,
        value=80.0,
        help="Insulin measurement.",
    )

    bmi = st.number_input(
        "BMI",
        min_value=0.0,
        max_value=80.0,
        value=25.0,
        help="Body Mass Index.",
    )

    pedigree = st.number_input(
        "Diabetes pedigree function",
        min_value=0.0,
        max_value=3.0,
        value=0.5,
        help="A value representing family-history-related diabetes risk.",
    )

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=30,
        help="Patient age in years.",
    )


predict_button = st.button(
    "Estimate risk",
    type="primary",
    use_container_width=True,
)


if predict_button:
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
    prediction = int(model.predict(input_data)[0])

    st.divider()
    st.subheader("Model result")

    metric_column, message_column = st.columns([1, 2])

    with metric_column:
        st.metric(
            label="Estimated probability",
            value=f"{probability:.1%}",
        )

    with message_column:
        if prediction == 1:
            st.error(
                "The model classified this input as higher risk."
            )
        else:
            st.success(
                "The model classified this input as lower risk."
            )

    st.progress(probability)

    st.caption(
        """
        The result reflects patterns learned from the training dataset.
        It is not a diagnosis and should not be interpreted as medical advice.
        """
    )

    with st.expander("View submitted values"):
        st.dataframe(
            input_data,
            use_container_width=True,
            hide_index=True,
        )

    st.divider()
    st.subheader("Model feature importance")

    model_step = model.named_steps["model"]

    if hasattr(model_step, "feature_importances_"):
        feature_names = input_data.columns
        importance_values = model_step.feature_importances_

        importance_df = pd.DataFrame(
            {
                "Feature": feature_names,
                "Importance": importance_values,
            }
        ).sort_values(
            by="Importance",
            ascending=False,
        )

        st.bar_chart(
            importance_df.set_index("Feature")
        )

        most_important_feature = importance_df.iloc[0]["Feature"]

        st.info(
            f"The model's most influential feature is "
            f"**{most_important_feature}**."
        )

        st.caption(
            "Feature importance shows which inputs were most useful to the model. "
            "It does not prove that a feature caused diabetes."
        )

    else:
        st.info(
            "Feature importance is not available for the selected model."
        )
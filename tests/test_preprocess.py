import pandas as pd
from src.data import clean_dataset


def test_zero_values_become_missing():
    df = pd.DataFrame({
        "Glucose": [0, 120],
        "BloodPressure": [70, 0],
        "SkinThickness": [20, 30],
        "Insulin": [80, 0],
        "BMI": [25.0, 0.0],
    })

    cleaned = clean_dataset(df)

    assert pd.isna(cleaned.loc[0, "Glucose"])
    assert pd.isna(cleaned.loc[1, "BloodPressure"])
    assert pd.isna(cleaned.loc[1, "Insulin"])
    assert pd.isna(cleaned.loc[1, "BMI"])

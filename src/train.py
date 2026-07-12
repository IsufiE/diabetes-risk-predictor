from pathlib import Path

import joblib
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data import clean_dataset, load_dataset


DATA_PATH = Path("data/diabetes.csv")
MODEL_PATH = Path("models/diabetes_model.joblib")


def logistic_pipeline(columns):
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    preprocess = ColumnTransformer([
        ("numeric", numeric, columns)
    ])

    return Pipeline([
        ("preprocess", preprocess),
        (
            "model",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ])


def forest_pipeline(columns):
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    preprocess = ColumnTransformer([
        ("numeric", numeric, columns)
    ])

    return Pipeline([
        ("preprocess", preprocess),
        (
            "model",
            RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ])


def evaluate(name, model, x_test, y_test):
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]

    auc = roc_auc_score(y_test, probabilities)

    print(f"\n{name}")
    print("=" * len(name))
    print(f"ROC-AUC: {auc:.3f}")

    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification report:")
    print(classification_report(
        y_test,
        predictions,
        zero_division=0,
    ))

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        display_labels=["No Diabetes", "Diabetes"],
    )

    plt.title(f"{name} Confusion Matrix")
    plt.tight_layout()
    plt.show()

    return auc


def main():
    df = clean_dataset(load_dataset(DATA_PATH))

    print(df.head())
    print()

    df.info()
    print()

    print(df.describe())

    x = df.drop(columns="Outcome")
    y = df["Outcome"].astype(int)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    models = {
        "Logistic Regression": logistic_pipeline(list(x.columns)),
        "Random Forest": forest_pipeline(list(x.columns)),
    }

    scores = {}

    for name, model in models.items():
        model.fit(x_train, y_train)
        scores[name] = evaluate(
            name,
            model,
            x_test,
            y_test,
        )

    best_name = max(scores, key=scores.get)

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(models[best_name], MODEL_PATH)

    print(f"\nSaved best model: {best_name} -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
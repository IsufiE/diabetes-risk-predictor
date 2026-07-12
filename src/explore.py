from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from src.data import clean_dataset, load_dataset

DATA_PATH = Path("data/diabetes.csv")


def main():
    df = clean_dataset(load_dataset(DATA_PATH))

    print("Dataset shape:")
    print(df.shape)

    print("\nClass distribution:")
    print(df["Outcome"].value_counts())

    print("\nClass percentages:")
    print(df["Outcome"].value_counts(normalize=True) * 100)

    average_values = df.groupby("Outcome")[["Glucose", "BMI", "Age"]].mean()

    ax = average_values.plot(
        kind="bar",
        figsize=(8, 6),
        width=0.75
    )

    ax.set_title("Average Patient Measurements by Diabetes Outcome")
    ax.set_xlabel("Outcome (0 = No Diabetes, 1 = Diabetes)")
    ax.set_ylabel("Average Value")

    plt.xticks(rotation=0)
    plt.legend(title="Feature")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 8))

    sns.heatmap(
        df.corr(numeric_only=True),
        annot=True,
        cmap="coolwarm",
        fmt=".2f"
    )

    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
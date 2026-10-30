
from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

# -----------------------------
# 1. Project paths and settings
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "reviews.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "sentiment_model.joblib"
REPORT_PATH = MODEL_DIR / "evaluation_report.json"

TEXT_COLUMN = "review"
LABEL_COLUMN = "sentiment"

VALID_LABELS = {"positive", "neutral", "negative"}

# -----------------------------
# 2. Load and validate dataset
# -----------------------------
def load_dataset():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Place reviews.csv inside the data folder."
        )

    df = pd.read_csv(DATA_PATH)

    required_columns = {TEXT_COLUMN, LABEL_COLUMN}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing columns: {sorted(missing_columns)}. "
            "Expected columns: review, sentiment"
        )

    df = df[[TEXT_COLUMN, LABEL_COLUMN]].copy()
    df[TEXT_COLUMN] = df[TEXT_COLUMN].fillna("").astype(str).str.strip()
    df[LABEL_COLUMN] = (
        df[LABEL_COLUMN].fillna("").astype(str).str.strip().str.lower()
    )

    df = df[
        (df[TEXT_COLUMN] != "")
        & (df[LABEL_COLUMN].isin(VALID_LABELS))
    ]

    # Remove exact duplicate review-label pairs.
    df = df.drop_duplicates(subset=[TEXT_COLUMN, LABEL_COLUMN])

    # Detect reviews that have conflicting sentiment labels.
    label_counts = df.groupby(TEXT_COLUMN)[LABEL_COLUMN].nunique()
    conflicting_reviews = label_counts[label_counts > 1].index

    if len(conflicting_reviews) > 0:
        print(
            f"Warning: removing {len(conflicting_reviews)} reviews "
            "that have conflicting labels."
        )
        df = df[~df[TEXT_COLUMN].isin(conflicting_reviews)]

    if df.empty:
        raise ValueError("No valid labeled reviews remain in the dataset.")

    counts = df[LABEL_COLUMN].value_counts()
    print("\nDataset class counts:")
    print(counts.to_string())

    missing_labels = VALID_LABELS - set(df[LABEL_COLUMN])

    if missing_labels:
        raise ValueError(
            f"Dataset is missing these classes: {sorted(missing_labels)}"
        )

    if counts.min() < 2:
        raise ValueError(
            "Each sentiment class needs at least two examples."
        )

    return df


# -----------------------------
# 3. Train and evaluate
# -----------------------------
def main():
    df = load_dataset()

    X = df[TEXT_COLUMN]
    y = df[LABEL_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                max_features=30000,
                sublinear_tf=True,
                strip_accents="unicode",
            ),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ])

    print("\nTraining TF-IDF + Logistic Regression...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0,
    )

    report = classification_report(
        y_test,
        predictions,
        labels=["negative", "neutral", "positive"],
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=["negative", "neutral", "positive"],
    )

    print("\n" + "=" * 45)
    print("TF-IDF MODEL EVALUATION")
    print("=" * 45)
    print(f"Training examples: {len(X_train)}")
    print(f"Test examples:    {len(X_test)}")
    print(f"Accuracy:         {accuracy:.4f}")
    print(f"Macro F1:         {macro_f1:.4f}")

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=["negative", "neutral", "positive"],
            zero_division=0,
        )
    )

    print("Confusion matrix")
    print("Class order: negative, neutral, positive")
    print(matrix)

    # -----------------------------
    # 4. Save model and report
    # -----------------------------
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    evaluation = {
        "model": "TF-IDF + Logistic Regression",
        "dataset_rows_after_cleaning": int(len(df)),
        "training_examples": int(len(X_train)),
        "test_examples": int(len(X_test)),
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "class_order": ["negative", "neutral", "positive"],
        "confusion_matrix": matrix.tolist(),
        "classification_report": report,
        "warning": (
            "Metrics are only as representative as the dataset. "
            "A random split is not independent real-world validation."
        ),
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        json.dump(evaluation, file, indent=4)

    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"Evaluation report saved to: {REPORT_PATH}")
    print("\nTraining and evaluation complete.")


if __name__ == "__main__":
    main()

"""Train/load the notebook's selected classifier and evaluate its held-out test set."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent
TRAIN_PATH = ROOT / "Dataset" / "train_split.csv"
TEST_PATH = ROOT / "Dataset" / "test_split.csv"
COMPARISON_PATH = ROOT / "model_comparison.csv"
MODEL_PATH = ROOT / "model.pkl"
TARGET = "y"


def load_split(path: Path):
    """Load a prepared split and separate the target from the input features."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path.name} is missing. Run the notebook's split/export cells first."
        )
    data = pd.read_csv(path)
    if TARGET not in data.columns:
        raise ValueError(f"Expected target column {TARGET!r} in {path}.")
    X_part = data.drop(columns=[TARGET])
    target_values = data[TARGET]
    if pd.api.types.is_numeric_dtype(target_values):
        # The notebook exports its encoded target as integers 0 and 1.
        y_part = target_values.astype(int)
    else:
        # Also accept the original dataset's yes/no labels.
        y_part = (target_values.astype(str).str.lower() == "yes").astype(int)
    if not set(y_part.unique()).issubset({0, 1}):
        raise ValueError(f"Target in {path} must contain only yes/no or 0/1 values.")
    return X_part, y_part


def build_notebook_model(X_train: pd.DataFrame) -> Pipeline:
    """Recreate the notebook preprocessing and its top PR-AUC model."""
    numeric_features = X_train.select_dtypes(include="number").columns.tolist()
    categorical_features = X_train.select_dtypes(exclude="number").columns.tolist()

    numeric_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessing = ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        sparse_threshold=0,
    )
    classifier = HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.1,
        random_state=42,
    )
    return Pipeline([("prep", preprocessing), ("model", classifier)])


def main() -> None:
    X_train, y_train = load_split(TRAIN_PATH)
    X_test, y_test = load_split(TEST_PATH)

    if list(X_test.columns) != list(X_train.columns):
        raise ValueError("Training and test splits do not have matching feature columns.")

    print("Model: HistGradientBoostingClassifier (best test PR-AUC model in notebook)")
    print(f"Dataset splits: {TRAIN_PATH.name} and {TEST_PATH.name}")
    print(f"Train rows: {len(X_train):,} | Test rows: {len(X_test):,}")
    print(f"Target: {TARGET} (yes=1, no=0) | Input features: {X_train.shape[1]}")

    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
        print(f"Loaded saved model: {MODEL_PATH.name}")
    else:
        model = build_notebook_model(X_train)
        model.fit(X_train, y_train)
        joblib.dump(model, MODEL_PATH)
        print(f"Trained model using the notebook's training split and saved: {MODEL_PATH.name}")

    predictions = model.predict(X_test)
    print("\nHeld-out test metrics")
    print(f"Accuracy:  {accuracy_score(y_test, predictions):.4f}")
    print(f"Precision: {precision_score(y_test, predictions, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_test, predictions, zero_division=0):.4f}")
    print(f"F1:        {f1_score(y_test, predictions, zero_division=0):.4f}")
    print("Confusion matrix (rows=true [no, yes], columns=predicted [no, yes]):")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))

    sample_predictions = model.predict(X_test.iloc[:5])
    true_labels = np.where(y_test.iloc[:5].to_numpy() == 1, "yes", "no")
    predicted_labels = np.where(sample_predictions == 1, "yes", "no")
    print("\nPredictions for the first five held-out test rows")
    print("row | true y | predicted y")
    for row_number, (true_value, predicted_value) in enumerate(
        zip(true_labels, predicted_labels), start=1
    ):
        print(f"{row_number:>3} | {true_value:>6} | {predicted_value:>11}")


if __name__ == "__main__":
    main()

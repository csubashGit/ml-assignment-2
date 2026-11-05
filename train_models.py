"""Train five classifiers on the UCI Obesity Levels dataset and save them to model/.

Usage: python train_models.py
"""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

SEED = 27
ROOT = Path(__file__).parent
DATA_FILE = ROOT / "data" / "ObesityDataSet_raw_and_data_sinthetic.csv"
MODEL_DIR = ROOT / "model"
TARGET = "NObeyesdad"

CATEGORICAL_COLS = [
    "Gender", "family_history_with_overweight", "FAVC", "CAEC",
    "SMOKE", "SCC", "CALC", "MTRANS",
]
NUMERIC_COLS = ["Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE"]


def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_COLS),
    ])


def candidate_models() -> dict:
    return {
        "Logistic Regression": LogisticRegression(max_iter=3000, C=5.0, random_state=SEED),
        "Decision Tree": DecisionTreeClassifier(max_depth=12, min_samples_leaf=2, random_state=SEED),
        "KNN": KNeighborsClassifier(n_neighbors=7, weights="distance"),
        "Naive Bayes": GaussianNB(),
        "SVM": SVC(kernel="rbf", C=10.0, gamma="scale", probability=True, random_state=SEED),
    }


def evaluate(pipeline: Pipeline, X, y) -> dict:
    y_pred = pipeline.predict(X)
    y_proba = pipeline.predict_proba(X)
    return {
        "Accuracy": accuracy_score(y, y_pred),
        "AUC": roc_auc_score(y, y_proba, multi_class="ovr", average="weighted",
                             labels=pipeline.classes_),
        "Precision": precision_score(y, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y, y_pred, average="weighted", zero_division=0),
        "F1": f1_score(y, y_pred, average="weighted", zero_division=0),
        "MCC": matthews_corrcoef(y, y_pred),
    }


def file_name(model_name: str) -> str:
    return model_name.lower().replace(" ", "_") + ".joblib"


def main() -> None:
    df = pd.read_csv(DATA_FILE).drop_duplicates().reset_index(drop=True)
    X, y = df.drop(columns=[TARGET]), df[TARGET]
    print(f"Dataset after de-duplication: {df.shape[0]} rows, {X.shape[1]} features, "
          f"{y.nunique()} classes")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )

    MODEL_DIR.mkdir(exist_ok=True)
    results = []
    for name, estimator in candidate_models().items():
        pipeline = Pipeline([("prep", make_preprocessor()), ("clf", estimator)])
        pipeline.fit(X_train, y_train)
        joblib.dump(pipeline, MODEL_DIR / file_name(name))
        scores = evaluate(pipeline, X_test, y_test)
        results.append({"Model": name, **scores})
        print(f"{name:20s} " + "  ".join(f"{k}={v:.4f}" for k, v in scores.items()))

    metrics = pd.DataFrame(results).round(4)
    metrics.to_csv(MODEL_DIR / "metrics.csv", index=False)

    test_df = X_test.copy()
    test_df[TARGET] = y_test
    test_df.to_csv(ROOT / "test_data.csv", index=False)

    print("\n" + metrics.to_string(index=False))


if __name__ == "__main__":
    main()

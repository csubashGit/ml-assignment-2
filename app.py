"""Streamlit app: evaluate the pre-trained obesity-level classifiers on an uploaded CSV."""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

from train_models import TARGET, evaluate, file_name

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "model"
# Must match the filenames produced by train_models.py
MODEL_NAMES = ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "SVM"]

st.set_page_config(page_title="Obesity Level Classifier", page_icon="âš–ï¸", layout="wide")


@st.cache_resource  # Load each model once and reuse across reruns
def load_model(name: str):
    return joblib.load(MODEL_DIR / file_name(name))


st.title("âš–ï¸ Obesity Level Classification: Model Explorer")
st.caption("UCI dataset: Estimation of Obesity Levels Based on Eating Habits and Physical Condition")

with st.sidebar:
    st.header("1. Upload test data")
    uploaded = st.file_uploader("CSV with the 16 feature columns (+ optional target)", type="csv")
    sample_path = ROOT / "test_data.csv"
    # Offer the generated sample file for download if it exists
    if sample_path.exists():
        st.download_button("Download sample test_data.csv", sample_path.read_bytes(),
                           file_name="test_data.csv", mime="text/csv")
    st.header("2. Choose a model")
    chosen = st.selectbox("Classifier", MODEL_NAMES, index=4)

tab_eval, tab_compare = st.tabs(["Evaluate selected model", "Compare all models"])

with tab_compare:
    # Show the leaderboard saved during training
    metrics_file = MODEL_DIR / "metrics.csv"
    if metrics_file.exists():
        st.subheader("Hold-out test metrics (from training run)")
        board = pd.read_csv(metrics_file).set_index("Model")
        st.dataframe(board.style.highlight_max(axis=0, color="#c6efce").format("{:.4f}"),
                     width="stretch")
        st.bar_chart(board)

with tab_eval:
    # Halt here until the user uploads a file
    if uploaded is None:
        st.info("Upload a CSV from the sidebar (you can use the sample test_data.csv).")
        st.stop()

    data = pd.read_csv(uploaded)
    st.write(f"Uploaded **{len(data)}** rows Ã— **{data.shape[1]}** columns")
    st.dataframe(data.head(), width="stretch")

    model = load_model(chosen)
    # Columns the model was trained on, in the expected order
    expected = list(model.feature_names_in_)
    missing = [c for c in expected if c not in data.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
        st.stop()

    features = data[expected]
    predictions = model.predict(features)

    # Without the target column we can predict but not score accuracy
    if TARGET not in data.columns:
        st.warning(f"No '{TARGET}' column found, so only predictions are shown.")
        st.dataframe(features.assign(Prediction=predictions), width="stretch")
        st.stop()

    actual = data[TARGET]
    scores = evaluate(model, features, actual)

    st.subheader(f"Evaluation metrics: {chosen}")
    cols = st.columns(len(scores))
    for col, (metric, value) in zip(cols, scores.items()):
        col.metric(metric, f"{value:.4f}")

    left, right = st.columns(2)
    with left:
        st.subheader("Confusion matrix")
        # Fix label order so rows/cols line up with the heatmap ticks
        labels = list(model.classes_)
        cm = confusion_matrix(actual, predictions, labels=labels)
        fig, ax = plt.subplots(figsize=(7, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu",
                    xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        st.pyplot(fig)
    with right:
        st.subheader("Classification report")
        report = classification_report(actual, predictions, output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(report).T.round(3), width="stretch")

    st.subheader("Predictions")
    st.dataframe(features.assign(Actual=actual, Predicted=predictions), width="stretch")

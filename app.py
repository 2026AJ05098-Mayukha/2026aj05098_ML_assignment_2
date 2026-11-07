"""Streamlit front-end: Breast Cancer Wisconsin (Diagnostic) classifier evaluation.

The five classifiers are trained offline by `train_models.py` and stored in `model/`.
This app only LOADS them: it scores an uploaded test CSV and reports metrics.
Nothing is ever re-trained on the uploaded data.
"""
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)

APP_DIR = Path(__file__).parent
MODEL_DIR = APP_DIR / "model"
SAMPLE_TEST_CSV = APP_DIR / "test_data.csv"
TARGET_COLUMN = "diagnosis"
CLASS_LABELS = {0: "Benign", 1: "Malignant"}

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}
METRIC_ORDER = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]

st.set_page_config(page_title="Tumour Diagnosis - Model Lab", page_icon="🔬", layout="wide")


# --------------------------------------------------------------------------- loading
@st.cache_resource(show_spinner=False)
def load_artifacts():
    feature_names = json.loads((MODEL_DIR / "feature_names.json").read_text())
    classifiers = {
        model_name: joblib.load(MODEL_DIR / artifact_name)
        for model_name, artifact_name in MODEL_FILES.items()
    }
    return feature_names, classifiers


@st.cache_data(show_spinner=False)
def load_training_time_metrics():
    path = MODEL_DIR / "metrics.csv"
    return pd.read_csv(path) if path.exists() else None


# --------------------------------------------------------------------------- helpers
def positive_score(classifier, feature_data):
    if hasattr(classifier, "predict_proba"):
        return classifier.predict_proba(feature_data)[:, 1]
    return classifier.decision_function(feature_data)


def prepare_upload(uploaded_data, feature_names):
    """Validate uploaded data and return features, labels, and validation notes."""
    uploaded_data = uploaded_data.copy()
    uploaded_data.columns = [str(column).strip() for column in uploaded_data.columns]

    if TARGET_COLUMN not in uploaded_data.columns:
        raise ValueError(f"The target column **`{TARGET_COLUMN}`** is missing from the uploaded file.")
    missing_features = [name for name in feature_names if name not in uploaded_data.columns]
    if missing_features:
        raise ValueError("Missing required feature column(s): "
                         + ", ".join(f"`{name}`" for name in missing_features))

    validation_notes = []
    extra_columns = [column for column in uploaded_data.columns
                     if column not in feature_names and column != TARGET_COLUMN]
    if extra_columns:
        validation_notes.append(f"Ignored {len(extra_columns)} extra column(s): "
                                f"{', '.join(extra_columns[:6])}"
                                + (" ..." if len(extra_columns) > 6 else ""))

    target_labels = uploaded_data[TARGET_COLUMN]
    if not pd.api.types.is_numeric_dtype(target_labels):  # accept M / B or full labels
        target_labels = target_labels.astype(str).str.strip().str.upper().str[0].map({"M": 1, "B": 0})
    uploaded_data[TARGET_COLUMN] = pd.to_numeric(target_labels, errors="coerce")
    for feature_name in feature_names:
        uploaded_data[feature_name] = pd.to_numeric(uploaded_data[feature_name], errors="coerce")

    original_row_count = len(uploaded_data)
    uploaded_data = uploaded_data.dropna(subset=feature_names + [TARGET_COLUMN])
    if len(uploaded_data) < original_row_count:
        validation_notes.append(
            f"Dropped {original_row_count - len(uploaded_data)} row(s) with missing / non-numeric values."
        )
    if len(uploaded_data) == 0:
        raise ValueError("No usable rows remain after cleaning.")
    if not set(uploaded_data[TARGET_COLUMN].unique()) <= {0, 1}:
        raise ValueError(f"`{TARGET_COLUMN}` must contain only 0 (benign) and 1 (malignant).")

    return (uploaded_data[feature_names], uploaded_data[TARGET_COLUMN].astype(int),
            validation_notes)


def compute_metrics(actual_labels, predicted_labels, positive_scores):
    try:
        auc = roc_auc_score(actual_labels, positive_scores)
    except ValueError:  # only one class present in the upload
        auc = np.nan
    return {
        "Accuracy": accuracy_score(actual_labels, predicted_labels),
        "AUC": auc,
        "Precision": precision_score(actual_labels, predicted_labels, zero_division=0),
        "Recall": recall_score(actual_labels, predicted_labels, zero_division=0),
        "F1": f1_score(actual_labels, predicted_labels, zero_division=0),
        "MCC": matthews_corrcoef(actual_labels, predicted_labels),
    }


def evaluate_all(classifiers, feature_data, actual_labels):
    metric_rows, prediction_cache = [], {}
    for model_name, classifier in classifiers.items():
        predicted_labels = classifier.predict(feature_data)
        positive_scores = positive_score(classifier, feature_data)
        prediction_cache[model_name] = (predicted_labels, positive_scores)
        metric_rows.append({
            "ML Model Name": model_name,
            **compute_metrics(actual_labels, predicted_labels, positive_scores),
        })
    return pd.DataFrame(metric_rows), prediction_cache


def format_metric(value):
    return "n/a" if pd.isna(value) else f"{value:.4f}"


# --------------------------------------------------------------------------- UI: sidebar
st.title("🔬 Tumour Diagnosis - Model Lab")
st.caption("Breast Cancer Wisconsin (Diagnostic) · five pre-trained classifiers · "
           "evaluation on your uploaded **test** data only (no re-training).")

try:
    FEATURE_NAMES, CLASSIFIERS = load_artifacts()
except Exception as exc:  # most common cause: scikit-learn version mismatch
    st.error("Could not load the saved models from `model/`.")
    st.exception(exc)
    st.info("Re-run `python train_models.py` with the same scikit-learn version as in "
            "`requirements.txt`, then redeploy.")
    st.stop()

with st.sidebar:
    st.header("1 · Test data")
    uploaded_file = st.file_uploader("Upload test CSV", type=["csv"],
                                     help=f"Must contain the {len(FEATURE_NAMES)} feature columns and a "
                                          f"`{TARGET_COLUMN}` column (1 = malignant, 0 = benign).")
    use_sample_data = st.checkbox("Use bundled sample (test_data.csv)", value=False,
                                  disabled=uploaded_file is not None)
    if SAMPLE_TEST_CSV.exists():
        st.download_button("⬇ Download sample test_data.csv", SAMPLE_TEST_CSV.read_bytes(),
                           file_name="test_data.csv", mime="text/csv")

    st.header("2 · Model")
    selected_model = st.selectbox("Choose a classifier", list(MODEL_FILES.keys()))

    st.divider()
    st.markdown(f"**Required columns**: {len(FEATURE_NAMES)} numeric features + `{TARGET_COLUMN}`")
    with st.expander("Show feature names"):
        st.write(FEATURE_NAMES)

# --------------------------------------------------------------------------- data intake
uploaded_data = None
if uploaded_file is not None:
    try:
        uploaded_data = pd.read_csv(uploaded_file)
    except Exception as exc:
        st.error(f"Could not read the CSV: {exc}")
        st.stop()
elif use_sample_data and SAMPLE_TEST_CSV.exists():
    uploaded_data = pd.read_csv(SAMPLE_TEST_CSV)

if uploaded_data is None:
    st.info("👈 Upload a test CSV (or tick the bundled sample) to begin.")
    ref = load_training_time_metrics()
    if ref is not None:
        st.subheader("Reference: metrics recorded when the models were trained")
        st.dataframe(ref.set_index("ML Model Name").style.format("{:.4f}"), width="stretch")
    st.stop()

try:
    feature_data, actual_labels, validation_notes = prepare_upload(uploaded_data, FEATURE_NAMES)
except ValueError as err:
    st.error(str(err))
    st.stop()

for validation_note in validation_notes:
    st.warning(validation_note)

row_count_column, malignant_count_column, benign_count_column = st.columns(3)
row_count_column.metric("Rows evaluated", len(feature_data))
malignant_count_column.metric("Malignant (1)", int((actual_labels == 1).sum()))
benign_count_column.metric("Benign (0)", int((actual_labels == 0).sum()))

all_metrics, prediction_cache = evaluate_all(CLASSIFIERS, feature_data, actual_labels)

selected_model_tab, comparison_tab, data_preview_tab = st.tabs(
    [f"📊 {selected_model}", "🏁 Compare all 5 models", "🗂 Data preview"]
)

# --------------------------------------------------------------------------- tab: selected model
with selected_model_tab:
    predicted_labels, positive_scores = prediction_cache[selected_model]
    selected_metrics = all_metrics.set_index("ML Model Name").loc[selected_model]

    st.subheader(f"Evaluation metrics - {selected_model}")
    metric_columns = st.columns(6)
    for metric_column, metric_name in zip(metric_columns, METRIC_ORDER):
        metric_column.metric(metric_name, format_metric(selected_metrics[metric_name]))

    matrix_column, roc_column = st.columns(2)
    with matrix_column:
        st.markdown("**Confusion matrix**")
        confusion_counts = confusion_matrix(actual_labels, predicted_labels, labels=[0, 1])
        figure, axis = plt.subplots(figsize=(4.6, 3.8))
        sns.heatmap(confusion_counts, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axis,
                    xticklabels=[CLASS_LABELS[0], CLASS_LABELS[1]],
                    yticklabels=[CLASS_LABELS[0], CLASS_LABELS[1]])
        axis.set_xlabel("Predicted")
        axis.set_ylabel("Actual")
        st.pyplot(figure)
        plt.close(figure)
        true_negative, false_positive, false_negative, true_positive = confusion_counts.ravel()
        st.caption(f"TN = {true_negative} · FP = {false_positive} · "
                   f"FN = {false_negative} · TP = {true_positive}  "
                   f"(a false negative means a malignant case was missed)")

    with roc_column:
        st.markdown("**ROC curve**")
        if actual_labels.nunique() == 2:
            false_positive_rate, true_positive_rate, _ = roc_curve(actual_labels, positive_scores)
            figure, axis = plt.subplots(figsize=(4.6, 3.8))
            axis.plot(false_positive_rate, true_positive_rate, lw=2,
                      label=f"AUC = {selected_metrics['AUC']:.4f}")
            axis.plot([0, 1], [0, 1], "--", color="grey", lw=1)
            axis.set_xlabel("False positive rate")
            axis.set_ylabel("True positive rate")
            axis.legend(loc="lower right")
            st.pyplot(figure)
            plt.close(figure)
        else:
            st.info("ROC curve needs both classes in the uploaded data.")

    st.markdown("**Classification report**")
    classification_summary = classification_report(
                                actual_labels, predicted_labels, labels=[0, 1],
                                target_names=[CLASS_LABELS[0], CLASS_LABELS[1]],
                                output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(classification_summary).T.round(4), width="stretch")

# --------------------------------------------------------------------------- tab: comparison
with comparison_tab:
    st.subheader("All five models on the uploaded test data")
    table = all_metrics.set_index("ML Model Name")[METRIC_ORDER]
    st.dataframe(table.style.format("{:.4f}", na_rep="n/a")
                 .highlight_max(axis=0, color="#cfe8cf"), width="stretch")
    st.caption("Green = best model for that metric.")

    best_model = table["MCC"].idxmax()
    st.success(f"Highest MCC on this data: **{best_model}** ({table.loc[best_model, 'MCC']:.4f})")

    chart_data = table.reset_index().melt(
        id_vars="ML Model Name", var_name="Metric", value_name="Score"
    )
    figure, axis = plt.subplots(figsize=(10, 4))
    sns.barplot(data=chart_data, x="Metric", y="Score", hue="ML Model Name", ax=axis)
    axis.set_ylim(max(0.0, np.nanmin(chart_data["Score"]) - 0.05), 1.01)
    axis.legend(loc="lower left", fontsize=8, ncol=3)
    st.pyplot(figure)
    plt.close(figure)

    st.download_button("⬇ Download comparison (CSV)", table.round(4).to_csv().encode(),
                       file_name="model_comparison.csv", mime="text/csv")

# --------------------------------------------------------------------------- tab: data
with data_preview_tab:
    st.dataframe(uploaded_data.head(50), width="stretch")
    st.caption(f"Showing first {min(50, len(uploaded_data))} of {len(uploaded_data)} uploaded rows.")
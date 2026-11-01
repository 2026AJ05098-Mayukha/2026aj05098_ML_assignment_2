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

HERE = Path(__file__).parent
MODEL_DIR = HERE / "model"
SAMPLE_CSV = HERE / "test_data.csv"
TARGET = "diagnosis"
CLASS_NAMES = {0: "Benign", 1: "Malignant"}

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
    features = json.loads((MODEL_DIR / "feature_names.json").read_text())
    models = {name: joblib.load(MODEL_DIR / fn) for name, fn in MODEL_FILES.items()}
    return features, models


@st.cache_data(show_spinner=False)
def load_training_time_metrics():
    path = MODEL_DIR / "metrics.csv"
    return pd.read_csv(path) if path.exists() else None


# --------------------------------------------------------------------------- helpers
def positive_score(pipe, X):
    if hasattr(pipe, "predict_proba"):
        return pipe.predict_proba(X)[:, 1]
    return pipe.decision_function(X)


def prepare_upload(df, features):
    """Validate the uploaded frame; return (X, y, notes) or raise ValueError."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    if TARGET not in df.columns:
        raise ValueError(f"The target column **`{TARGET}`** is missing from the uploaded file.")
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError("Missing required feature column(s): " + ", ".join(f"`{m}`" for m in missing))

    notes = []
    extra = [c for c in df.columns if c not in features and c != TARGET]
    if extra:
        notes.append(f"Ignored {len(extra)} extra column(s): {', '.join(extra[:6])}"
                     + (" ..." if len(extra) > 6 else ""))

    y = df[TARGET]
    if y.dtype == object:  # accept M / B (or Malignant / Benign) labels
        y = y.astype(str).str.strip().str.upper().str[0].map({"M": 1, "B": 0})
    df[TARGET] = pd.to_numeric(y, errors="coerce")
    for f in features:
        df[f] = pd.to_numeric(df[f], errors="coerce")

    before = len(df)
    df = df.dropna(subset=features + [TARGET])
    if len(df) < before:
        notes.append(f"Dropped {before - len(df)} row(s) with missing / non-numeric values.")
    if len(df) == 0:
        raise ValueError("No usable rows remain after cleaning.")
    if not set(df[TARGET].unique()) <= {0, 1}:
        raise ValueError(f"`{TARGET}` must contain only 0 (benign) and 1 (malignant).")

    return df[features], df[TARGET].astype(int), notes


def compute_metrics(y_true, y_pred, y_score):
    try:
        auc = roc_auc_score(y_true, y_score)
    except ValueError:  # only one class present in the upload
        auc = np.nan
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def evaluate_all(models, X, y):
    rows, cache = [], {}
    for name, pipe in models.items():
        pred = pipe.predict(X)
        score = positive_score(pipe, X)
        cache[name] = (pred, score)
        rows.append({"ML Model Name": name, **compute_metrics(y, pred, score)})
    return pd.DataFrame(rows), cache


def fmt(v):
    return "n/a" if pd.isna(v) else f"{v:.4f}"


# --------------------------------------------------------------------------- UI: sidebar
st.title("🔬 Tumour Diagnosis - Model Lab")
st.caption("Breast Cancer Wisconsin (Diagnostic) · five pre-trained classifiers · "
           "evaluation on your uploaded **test** data only (no re-training).")

try:
    FEATURES, MODELS = load_artifacts()
except Exception as exc:  # most common cause: scikit-learn version mismatch
    st.error("Could not load the saved models from `model/`.")
    st.exception(exc)
    st.info("Re-run `python train_models.py` with the same scikit-learn version as in "
            "`requirements.txt`, then redeploy.")
    st.stop()

with st.sidebar:
    st.header("1 · Test data")
    upload = st.file_uploader("Upload test CSV", type=["csv"],
                              help=f"Must contain the {len(FEATURES)} feature columns and a "
                                   f"`{TARGET}` column (1 = malignant, 0 = benign).")
    use_sample = st.checkbox("Use bundled sample (test_data.csv)", value=False,
                             disabled=upload is not None)
    if SAMPLE_CSV.exists():
        st.download_button("⬇ Download sample test_data.csv", SAMPLE_CSV.read_bytes(),
                           file_name="test_data.csv", mime="text/csv")

    st.header("2 · Model")
    chosen = st.selectbox("Choose a classifier", list(MODEL_FILES.keys()))

    st.divider()
    st.markdown(f"**Required columns**: {len(FEATURES)} numeric features + `{TARGET}`")
    with st.expander("Show feature names"):
        st.write(FEATURES)

# --------------------------------------------------------------------------- data intake
raw_df = None
if upload is not None:
    try:
        raw_df = pd.read_csv(upload)
    except Exception as exc:
        st.error(f"Could not read the CSV: {exc}")
        st.stop()
elif use_sample and SAMPLE_CSV.exists():
    raw_df = pd.read_csv(SAMPLE_CSV)

if raw_df is None:
    st.info("👈 Upload a test CSV (or tick the bundled sample) to begin.")
    ref = load_training_time_metrics()
    if ref is not None:
        st.subheader("Reference: metrics recorded when the models were trained")
        st.dataframe(ref.set_index("ML Model Name").style.format("{:.4f}"), width="stretch")
    st.stop()

try:
    X, y, notes = prepare_upload(raw_df, FEATURES)
except ValueError as err:
    st.error(str(err))
    st.stop()

for n in notes:
    st.warning(n)

c1, c2, c3 = st.columns(3)
c1.metric("Rows evaluated", len(X))
c2.metric("Malignant (1)", int((y == 1).sum()))
c3.metric("Benign (0)", int((y == 0).sum()))

all_metrics, preds = evaluate_all(MODELS, X, y)

tab_one, tab_all, tab_data = st.tabs([f"📊 {chosen}", "🏁 Compare all 5 models", "🗂 Data preview"])

# --------------------------------------------------------------------------- tab: selected model
with tab_one:
    y_pred, y_score = preds[chosen]
    m = all_metrics.set_index("ML Model Name").loc[chosen]

    st.subheader(f"Evaluation metrics - {chosen}")
    cols = st.columns(6)
    for col, key in zip(cols, METRIC_ORDER):
        col.metric(key, fmt(m[key]))

    left, right = st.columns(2)
    with left:
        st.markdown("**Confusion matrix**")
        cm = confusion_matrix(y, y_pred, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(4.6, 3.8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=[CLASS_NAMES[0], CLASS_NAMES[1]],
                    yticklabels=[CLASS_NAMES[0], CLASS_NAMES[1]])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
        plt.close(fig)
        tn, fp, fn, tp = cm.ravel()
        st.caption(f"TN = {tn} · FP = {fp} · FN = {fn} · TP = {tp}  "
                   f"(a false negative means a malignant case was missed)")

    with right:
        st.markdown("**ROC curve**")
        if y.nunique() == 2:
            fpr, tpr, _ = roc_curve(y, y_score)
            fig, ax = plt.subplots(figsize=(4.6, 3.8))
            ax.plot(fpr, tpr, lw=2, label=f"AUC = {m['AUC']:.4f}")
            ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
            ax.set_xlabel("False positive rate")
            ax.set_ylabel("True positive rate")
            ax.legend(loc="lower right")
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("ROC curve needs both classes in the uploaded data.")

    st.markdown("**Classification report**")
    rep = classification_report(y, y_pred, labels=[0, 1],
                                target_names=[CLASS_NAMES[0], CLASS_NAMES[1]],
                                output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(rep).T.round(4), width="stretch")

# --------------------------------------------------------------------------- tab: comparison
with tab_all:
    st.subheader("All five models on the uploaded test data")
    table = all_metrics.set_index("ML Model Name")[METRIC_ORDER]
    st.dataframe(table.style.format("{:.4f}", na_rep="n/a")
                 .highlight_max(axis=0, color="#cfe8cf"), width="stretch")
    st.caption("Green = best model for that metric.")

    best = table["MCC"].idxmax()
    st.success(f"Highest MCC on this data: **{best}** ({table.loc[best, 'MCC']:.4f})")

    long = table.reset_index().melt(id_vars="ML Model Name", var_name="Metric", value_name="Score")
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(data=long, x="Metric", y="Score", hue="ML Model Name", ax=ax)
    ax.set_ylim(max(0.0, np.nanmin(long["Score"]) - 0.05), 1.01)
    ax.legend(loc="lower left", fontsize=8, ncol=3)
    st.pyplot(fig)
    plt.close(fig)

    st.download_button("⬇ Download comparison (CSV)", table.round(4).to_csv().encode(),
                       file_name="model_comparison.csv", mime="text/csv")

# --------------------------------------------------------------------------- tab: data
with tab_data:
    st.dataframe(raw_df.head(50), width="stretch")
    st.caption(f"Showing first {min(50, len(raw_df))} of {len(raw_df)} uploaded rows.")

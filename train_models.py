"""Train and evaluate five classifiers on the Breast Cancer Wisconsin (Diagnostic) data.

Outputs
-------
model/<name>.joblib      fitted sklearn Pipeline (scaler + classifier) for each model
model/feature_names.json ordered list of the 30 input features
model/metrics.csv        evaluation metrics of every model on the hold-out test set
test_data.csv            hold-out test split (features + target) for the Streamlit app
full_dataset.csv         complete dataset (reference only)
"""
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef)

SEED = 2026
TARGET = "diagnosis"          # 1 = malignant, 0 = benign
ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "model"
MODEL_DIR.mkdir(exist_ok=True)


def load_data():
    raw = load_breast_cancer(as_frame=True)
    X = raw.data.copy()
    # sklearn encodes malignant=0 / benign=1; flip so that malignant (the class of
    # clinical interest) is the positive class (1).
    y = (1 - raw.target).rename(TARGET)
    return X, y


def build_models():
    return {
        "Logistic Regression": LogisticRegression(C=1.0, max_iter=2000, random_state=SEED),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, min_samples_leaf=3, random_state=SEED),
        "KNN": KNeighborsClassifier(n_neighbors=7, weights="distance"),
        "Naive Bayes": GaussianNB(),
        "Support Vector Machine": SVC(kernel="rbf", C=2.0, gamma="scale", random_state=SEED),
    }


def file_name(model_name):
    return model_name.lower().replace(" ", "_") + ".joblib"


def positive_score(pipe, X):
    """Continuous score for the positive class (used for AUC).

    Uses predict_proba when the model has it, otherwise the SVM decision_function
    (signed distance to the hyperplane), which gives the same ranking-based AUC.
    """
    if hasattr(pipe, "predict_proba"):
        return pipe.predict_proba(X)[:, 1]
    return pipe.decision_function(X)


def evaluate(y_true, y_pred, y_score):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_score),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def main():
    X, y = load_data()
    print(f"Dataset: {X.shape[0]} instances, {X.shape[1]} features, "
          f"class balance (0/1) = {y.value_counts().to_dict()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEED)

    pd.concat([X, y], axis=1).to_csv(ROOT / "full_dataset.csv", index=False)
    pd.concat([X_test, y_test], axis=1).to_csv(ROOT / "test_data.csv", index=False)
    (MODEL_DIR / "feature_names.json").write_text(json.dumps(list(X.columns), indent=2))

    rows = []
    for name, clf in build_models().items():
        pipe = Pipeline([("scale", StandardScaler()), ("clf", clf)])
        pipe.fit(X_train, y_train)
        joblib.dump(pipe, MODEL_DIR / file_name(name))

        pred = pipe.predict(X_test)
        score = positive_score(pipe, X_test)
        rows.append({"ML Model Name": name, **evaluate(y_test, pred, score)})

    metrics = pd.DataFrame(rows).round(4)
    metrics.to_csv(MODEL_DIR / "metrics.csv", index=False)
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()

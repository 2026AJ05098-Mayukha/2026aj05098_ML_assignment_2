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

RANDOM_SEED = 2026
TARGET_COLUMN = "diagnosis"  # 1 = malignant, 0 = benign
PROJECT_DIR = Path(__file__).parent
MODEL_DIR = PROJECT_DIR / "model"
MODEL_DIR.mkdir(exist_ok=True)


def load_data():
    dataset = load_breast_cancer(as_frame=True)
    feature_data = dataset.data.copy()
    # sklearn encodes malignant=0 / benign=1; flip so that malignant (the class of
    # clinical interest) is the positive class (1).
    target_labels = (1 - dataset.target).rename(TARGET_COLUMN)
    return feature_data, target_labels


def build_models():
    return {
        "Logistic Regression": LogisticRegression(C=1.0, max_iter=2000, random_state=RANDOM_SEED),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, min_samples_leaf=3, random_state=RANDOM_SEED),
        "KNN": KNeighborsClassifier(n_neighbors=7, weights="distance"),
        "Naive Bayes": GaussianNB(),
        "Support Vector Machine": SVC(kernel="rbf", C=2.0, gamma="scale", random_state=RANDOM_SEED),
    }


def artifact_filename(classifier_name):
    return classifier_name.lower().replace(" ", "_") + ".joblib"


def positive_score(model_pipeline, feature_data):
    """Continuous score for the positive class (used for AUC).

    Uses predict_proba when the model has it, otherwise the SVM decision_function
    (signed distance to the hyperplane), which gives the same ranking-based AUC.
    """
    if hasattr(model_pipeline, "predict_proba"):
        return model_pipeline.predict_proba(feature_data)[:, 1]
    return model_pipeline.decision_function(feature_data)


def evaluate(actual_labels, predicted_labels, positive_scores):
    return {
        "Accuracy": accuracy_score(actual_labels, predicted_labels),
        "AUC": roc_auc_score(actual_labels, positive_scores),
        "Precision": precision_score(actual_labels, predicted_labels),
        "Recall": recall_score(actual_labels, predicted_labels),
        "F1": f1_score(actual_labels, predicted_labels),
        "MCC": matthews_corrcoef(actual_labels, predicted_labels),
    }


def main():
    feature_data, target_labels = load_data()
    print(f"Dataset: {feature_data.shape[0]} instances, {feature_data.shape[1]} features, "
          f"class balance (0/1) = {target_labels.value_counts().to_dict()}")

    training_features, test_features, training_labels, test_labels = train_test_split(
        feature_data, target_labels, test_size=0.20, stratify=target_labels,
        random_state=RANDOM_SEED)

    pd.concat([feature_data, target_labels], axis=1).to_csv(
        PROJECT_DIR / "full_dataset.csv", index=False)
    pd.concat([test_features, test_labels], axis=1).to_csv(
        PROJECT_DIR / "test_data.csv", index=False)
    (MODEL_DIR / "feature_names.json").write_text(
        json.dumps(list(feature_data.columns), indent=2))

    metric_rows = []
    for classifier_name, classifier in build_models().items():
        model_pipeline = Pipeline([("scale", StandardScaler()), ("clf", classifier)])
        model_pipeline.fit(training_features, training_labels)
        joblib.dump(model_pipeline, MODEL_DIR / artifact_filename(classifier_name))

        predicted_labels = model_pipeline.predict(test_features)
        positive_scores = positive_score(model_pipeline, test_features)
        metric_rows.append({
            "ML Model Name": classifier_name,
            **evaluate(test_labels, predicted_labels, positive_scores),
        })

    metrics = pd.DataFrame(metric_rows).round(4)
    metrics.to_csv(MODEL_DIR / "metrics.csv", index=False)
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()

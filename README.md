# Breast Cancer Classification Model Lab

This project trains five scikit-learn classifiers on the Breast Cancer Wisconsin Diagnostic dataset and provides a Streamlit app for evaluating those saved models on labeled test data. Uploaded data is used for prediction and evaluation only; the app does not retrain models.

> This is an educational machine-learning project, not a medical diagnostic tool.

## Models

Every classifier is trained in a pipeline that applies `StandardScaler` before classification. The train/test split is stratified and uses a fixed random seed (`2026`).

| Classifier | Configuration |
|---|---|
| Logistic Regression | `C=1.0`, `max_iter=2000` |
| Decision Tree | `max_depth=5`, `min_samples_leaf=3` |
| K-Nearest Neighbors | `n_neighbors=7`, distance weighting |
| Gaussian Naive Bayes | Default configuration |
| Support Vector Machine | RBF kernel, `C=2.0`, `gamma="scale"` |

The target column is named `diagnosis`. Labels use `1` for malignant and `0` for benign, so malignant is treated as the positive class for evaluation.

## Run Locally

Use Python with the packages listed in `requirements.txt`:

```bash
python -m pip install -r requirements.txt
python train_models.py
streamlit run app.py
```

Running `train_models.py` loads the dataset distributed with scikit-learn, makes a stratified 80/20 split, trains the classifiers, and writes the generated data and model artifacts. The app expects the `model/` artifacts; rerun the training script if they need to be regenerated or if the installed scikit-learn version is incompatible with the saved models.

## Streamlit App

In the sidebar, upload a labeled test CSV or choose the bundled `test_data.csv`, then select a classifier. The app validates the required columns, converts numeric feature values, drops rows with missing or non-numeric required values, and ignores additional columns.

An upload must contain all 30 numeric feature columns used by the training dataset and a `diagnosis` column. Target labels may be `0`/`1` or `B`/`M` (including `Benign`/`Malignant`). The included `test_data.csv` is an example of the expected format.

For the selected classifier, the app displays accuracy, ROC AUC, precision, recall, F1, Matthews correlation coefficient (MCC), a confusion matrix, a classification report, and a ROC curve when both classes are present. A separate tab compares all five classifiers on the same uploaded rows and allows the metrics to be downloaded as CSV.

## Project Files

| File or directory | Purpose |
|---|---|
| `app.py` | Streamlit interface, upload validation, predictions, and evaluation charts |
| `train_models.py` | Loads data, trains classifiers, and saves evaluation results and artifacts |
| `test_data.csv` | Labeled hold-out test split used as the bundled app example |
| `full_dataset.csv` | Full dataset exported by the training script |
| `model/*.joblib` | Fitted preprocessing and classifier pipelines |
| `model/feature_names.json` | Ordered list of input feature names expected by the app |
| `model/metrics.csv` | Hold-out metrics generated during training |
| `requirements.txt` | Python package versions for the project |

## Metrics and Interpretation

Training metrics are calculated on the held-out split and saved to `model/metrics.csv`. The app calculates the same metrics for the selected uploaded test data. ROC AUC uses predicted probabilities where available and the SVM decision score otherwise. When an uploaded file contains only one target class, AUC is unavailable and the ROC curve is omitted.

Metric results describe only the supplied dataset and split. In particular, recall measures how many malignant examples were identified, while a false negative is a malignant example predicted as benign. These results should not be interpreted as evidence of clinical performance.
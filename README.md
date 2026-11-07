# Breast Cancer Classification Model Lab

This project trains five scikit-learn classifiers on the Breast Cancer Wisconsin Diagnostic dataset and provides a Streamlit app for evaluating those saved models on labeled test data. Uploaded data is used for prediction and evaluation only; the app does not retrain models.

This is an educational machine-learning project, not a medical diagnostic tool.

## a. Problem statement

Telling malignant breast tumours apart from benign ones early and accurately is a high-stakes classification task. Each sample has 30 numeric measurements computed from a digitised image of a fine-needle aspirate (FNA) of a breast mass, and the goal is to predict whether the tumour is malignant (1) or benign (0).

Five classifiers (Logistic Regression, Decision Tree, K-Nearest Neighbors, Gaussian Naive Bayes and Support Vector Machine) are trained on the same data and compared on six metrics: Accuracy, AUC, Precision, Recall, F1 and MCC. The trained models are served through a Streamlit app that evaluates any uploaded test CSV without retraining.

A missed cancer (false negative) is more costly than a false alarm, so recall on the malignant class is read alongside the overall metrics when picking a winner.

## b. Dataset description

| Item | Value |
|---|---|
| Name | Breast Cancer Wisconsin (Diagnostic) |
| Source | UCI Machine Learning Repository, loaded through scikit-learn (`load_breast_cancer`) |
| Problem type | Binary classification |
| Instances | 569 (requirement: at least 500) |
| Features | 30 numeric (requirement: at least 12) |
| Target | `diagnosis`: 1 = malignant (212), 0 = benign (357) |
| Missing values | None |

The 30 features are the mean, standard error and "worst" (mean of the three largest values) of 10 cell-nucleus characteristics: radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry and fractal dimension.

### Pre-processing and split

- scikit-learn encodes malignant as 0, so the label is flipped to make malignant the positive class (1).
- A stratified 80/20 train-test split with `random_state = 2026` gives 455 training rows and 114 test rows. The test split is saved as `test_data.csv`.
- `StandardScaler` is part of every model's pipeline and is fitted on training data only, so the saved `.joblib` files already include preprocessing.

## c. GitHub repository link

GitHub: https://github.com/2026AJ05098-Mayukha/2026aj05098_ML_assignment_2

Live Streamlit app: https://ml-assignment2-2026aj05098.streamlit.app/

## d. Models used

Every classifier is trained in a pipeline that applies `StandardScaler` before classification.

| Classifier | Configuration |
|---|---|
| Logistic Regression | C=1.0, max_iter=2000 |
| Decision Tree | max_depth=5, min_samples_leaf=3 |
| K-Nearest Neighbors | n_neighbors=7, distance weighting |
| Gaussian Naive Bayes | Default configuration |
| Support Vector Machine | RBF kernel, C=2.0, gamma="scale" |

Comparison table: metrics on the 114-row hold-out test set (positive class = malignant).

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9737 | 0.9987 | 0.9756 | 0.9524 | 0.9639 | 0.9433 |
| Decision Tree | 0.9737 | 0.9785 | 0.9756 | 0.9524 | 0.9639 | 0.9433 |
| KNN | 0.9737 | 0.9970 | 0.9535 | 0.9762 | 0.9647 | 0.9439 |
| Naive Bayes | 0.9561 | 0.9964 | 0.9302 | 0.9524 | 0.9412 | 0.9064 |
| Support Vector Machine | 0.9825 | 1.0000 | 0.9545 | 1.0000 | 0.9767 | 0.9633 |

Confusion-matrix counts behind these numbers:

| Model | TN | FP | FN | TP |
|---|---|---|---|---|
| Logistic Regression | 71 | 1 | 2 | 40 |
| Decision Tree | 71 | 1 | 2 | 40 |
| KNN | 70 | 2 | 1 | 41 |
| Naive Bayes | 69 | 3 | 2 | 40 |
| Support Vector Machine | 70 | 2 | 0 | 42 |

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Strong linear baseline with 97.37% accuracy and an AUC of 0.9987, so the two classes are close to linearly separable once features are standardised. It has the joint-highest precision (0.9756) but misses 2 of the 42 malignant cases (recall 0.9524). |
| Decision Tree | Matches Logistic Regression on every threshold-based metric (same confusion matrix), but its AUC of 0.9785 is the lowest of the five. A depth-5 tree gives coarse leaf probabilities, so it ranks cases less finely, and tree results tend to shift with a different split. |
| KNN | Same 97.37% accuracy, but it catches one more malignant case than Logistic Regression (recall 0.9762, 1 miss), at the cost of 2 false alarms (precision 0.9535). It relies on the scaling done in the pipeline and must keep the whole training set for prediction. |
| Naive Bayes | The weakest model on accuracy (95.61%), precision (0.9302) and MCC (0.9064), with the most false positives (3). The independence assumption is strained because many features (radius, perimeter, area) are strongly correlated. Its AUC (0.9964) is still competitive and it is the cheapest to train. |
| Support Vector Machine | Best on accuracy (98.25%), AUC (1.0000), recall, F1 and MCC. It flagged all 42 malignant samples (0 false negatives) with 2 false positives, so the RBF kernel handles the slightly non-linear boundary well. Its precision (0.9545) is a little below Logistic Regression and Decision Tree. |
| Overall winner for this dataset | **Support Vector Machine (RBF).** It leads on five of six metrics and misses no malignant case, which matters most for a screening task. Caveat: the test set has only 114 rows, so one sample shifts accuracy by about 0.9 percentage points, and the gaps between SVM, KNN and Logistic Regression are small. Cross-validation would be needed before drawing strong conclusions. |

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

An upload must contain all 30 numeric feature columns used by the training dataset and a `diagnosis` column. Target labels may be 0/1 or B/M (including Benign/Malignant). The included `test_data.csv` is an example of the expected format.

For the selected classifier, the app displays accuracy, ROC AUC, precision, recall, F1, Matthews correlation coefficient (MCC), a confusion matrix, a classification report, and a ROC curve when both classes are present. A separate tab compares all five classifiers on the same uploaded rows and allows the metrics to be downloaded as CSV.

## Project Files

| File or directory | Purpose |
|---|---|
| app.py | Streamlit interface, upload validation, predictions, and evaluation charts |
| train_models.py | Loads data, trains classifiers, and saves evaluation results and artifacts |
| test_data.csv | Labeled hold-out test split used as the bundled app example |
| full_dataset.csv | Full dataset exported by the training script |
| model/*.joblib | Fitted preprocessing and classifier pipelines |
| model/feature_names.json | Ordered list of input feature names expected by the app |
| model/metrics.csv | Hold-out metrics generated during training |
| requirements.txt | Python package versions for the project |

## Metrics and Interpretation

Training metrics are calculated on the held-out split and saved to `model/metrics.csv`. The app calculates the same metrics for the selected uploaded test data. ROC AUC uses predicted probabilities where available and the SVM decision score otherwise. When an uploaded file contains only one target class, AUC is unavailable and the ROC curve is omitted.

Metric results describe only the supplied dataset and split. In particular, recall measures how many malignant examples were identified, while a false negative is a malignant example predicted as benign. These results should not be interpreted as evidence of clinical performance.

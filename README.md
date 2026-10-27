# Heart Disease Classification: Machine Learning Benchmark & Streamlit Deployment

## a. Problem statement
Cardiovascular diseases are the leading cause of death worldwide, and early identification of high-risk patients allows timely clinical intervention. The goal of this project is to train and compare five supervised classification models that predict the presence or absence of heart disease from routine clinical measurements. The trained models are packaged in an interactive Streamlit web application that evaluates them on uploaded test data without retraining.

## b. Dataset description
- **Dataset Name:** UCI Heart Disease (combined: Cleveland, Hungary, Switzerland, VA Long Beach)
- **Source:** https://archive.ics.uci.edu/dataset/45/heart+disease (`processed.*.data` files)
- **Instances:** 920 patients (Cleveland 303, Hungary 294, VA 200, Switzerland 123)
- **Features:** 13 clinical attributes
- **Problem Type:** Binary classification
- **Target Variable:** `target` derived from the original `num` column (0 = no disease, 1–4 = disease → `target = 1`). Class balance: 411 negative / 509 positive.
- **Train/Test Split:** 80/20 stratified split (736 train, 184 test). The test split is saved as `test_data.csv`.

**Feature definitions:**
1. `age`: Age in years
2. `sex`: Sex (1 = male, 0 = female)
3. `cp`: Chest pain type (1: typical angina, 2: atypical angina, 3: non-anginal pain, 4: asymptomatic)
4. `trestbps`: Resting blood pressure (mm Hg)
5. `chol`: Serum cholesterol (mg/dl)
6. `fbs`: Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)
7. `restecg`: Resting ECG result (0: normal, 1: ST-T wave abnormality, 2: left ventricular hypertrophy)
8. `thalach`: Maximum heart rate achieved
9. `exang`: Exercise-induced angina (1 = yes, 0 = no)
10. `oldpeak`: ST depression induced by exercise relative to rest
11. `slope`: Slope of the peak exercise ST segment (1: upsloping, 2: flat, 3: downsloping)
12. `ca`: Number of major vessels (0–3) coloured by fluoroscopy
13. `thal`: Thallium stress test result (3: normal, 6: fixed defect, 7: reversible defect)

**Data quality and preprocessing:**
- Missing values are marked `?` in the source files. The most affected columns are `ca` (611 missing, 66%), `thal` (486, 53%) and `slope` (309, 34%). Missingness depends strongly on the hospital: these tests were mostly recorded only at Cleveland.
- 172 records have `chol = 0`, which is physiologically impossible; these (and any `trestbps = 0`) were treated as missing.
- All 13 features were kept (dropping `ca` and `thal` would fall below the 12-feature minimum). Each model pipeline uses `SimpleImputer(strategy="median", add_indicator=True)`, which fills gaps and adds a binary "was missing" flag per affected column, so the models can learn from the missingness pattern.
- Features are standardised with `StandardScaler` for all models except the Decision Tree.
- Hyperparameters were tuned with 5-fold `GridSearchCV` (scoring: ROC AUC) on the training set only.

## c. Github Repository Link
- **GitHub Repository:** https://github.com/2026aj05707/ML_Assignment
- **Live Streamlit App:** https://mlassignment-gnqwmgawx9d7ibsvh7b9u3.streamlit.app

## d. Models used:

| Model | Selected hyperparameters |
| :--- | :--- |
| Logistic Regression | `C = 0.03` |
| Decision Tree | `max_depth = 4`, `min_samples_leaf = 10` |
| KNN | `n_neighbors = 41`, `weights = "distance"` |
| Naive Bayes (Gaussian) | `var_smoothing = 1e-05` |
| Support Vector Machine (RBF) | `C = 1`, `gamma = "scale"` |

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Logistic Regression | 0.8207 | 0.8997 | 0.8485 | 0.8235 | 0.8358 | 0.6387 |
| Decision Tree | 0.7663 | 0.8569 | 0.8391 | 0.7157 | 0.7725 | 0.5425 |
| KNN | 0.8424 | 0.8998 | 0.8614 | 0.8529 | 0.8571 | 0.6814 |
| Naive Bayes | 0.6196 | 0.8570 | 0.8478 | 0.3824 | 0.5270 | 0.3409 |
| Support Vector Machine | 0.8315 | 0.9030 | 0.8447 | 0.8529 | 0.8488 | 0.6586 |

### Observations on Model Performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Gives a solid, balanced baseline across all metrics because the standardized clinical features separate fairly well linearly. |
| **Decision Tree** | Simple to interpret, but it has the lowest precision and struggles to catch as many positive cases as Logistic Regression, KNN and SVM. |
| **KNN** | Delivers the highest overall accuracy and MCC by grouping patients with similar overall clinical profiles together. |
| **Naive Bayes** | Struggles heavily on recall and overall accuracy because its assumption that all symptoms are independent does not hold for related heart indicators. |
| **Support Vector Machine** | Achieves the highest AUC and matches KNN on recall, using its curved boundary to separate complex patient cases smoothly. |
| **Overall, Winner for your dataset?** | **KNN (K-Nearest Neighbors)** is the overall winner because it scores highest in Accuracy (0.8424), Precision (0.8614), F1 (0.8571), and MCC (0.6814) while matching the top Recall (0.8529). |

## Project Structure

```
ML_Assignment/
├── app.py                  # Streamlit dashboard (upload, model selection, metrics, confusion matrix, model comparison)
├── ML_Assignment2.ipynb    # Data loading, cleaning, model tuning, evaluation and export
├── requirements.txt        # Python dependencies
├── test_data.csv           # Held-out test split (184 rows, 13 features + target)
└── model/                  # Saved scikit-learn pipelines (imputer + scaler + classifier)
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    └── support_vector_machine.joblib
```

## Running Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Upload `test_data.csv` from the sidebar (or download it via the sidebar button), pick a classifier, and the app evaluates the saved model on the uploaded data without retraining.

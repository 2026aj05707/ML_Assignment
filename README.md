# Heart Disease Classification: Machine Learning Benchmark & Streamlit Deployment

## a. Problem statement
Cardiovascular diseases (CVDs) remain the leading cause of mortality globally, accounting for an estimated 17.9 million deaths annually. Early and accurate detection of high-risk factors enables timely clinical interventions, reducing critical cardiac events. The primary objective of this project is to develop, evaluate, and benchmark five distinct supervised machine learning classification algorithms on clinical patient records to predict the presence or absence of heart disease. The finalized models are bundled into an interactive, cloud-deployed Streamlit web application capable of performing instant on-demand inference without retraining.

## b. Dataset description
- **Dataset Name:** UCI Heart Disease (Cleveland & Statlog clinical benchmarks)
- **Problem Classification:** Binary Classification
- **Target Variable:** `target` (0 = Absence of heart disease, 1 = Presence of heart disease)
- **Features Count:** 13 clinical attributes
- **Clinical Feature Definitions:**
  1. `age`: Age in years
  2. `sex`: Biological sex (1 = Male, 0 = Female)
  3. `cp`: Chest pain type (0: typical angina, 1: atypical angina, 2: non-anginal pain, 3: asymptomatic)
  4. `trestbps`: Resting blood pressure (in mm Hg on hospital admission)
  5. `chol`: Serum cholesterol in mg/dl
  6. `fbs`: Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)
  7. `restecg`: Resting electrocardiographic measurement (0: normal, 1: ST-T wave abnormality, 2: left ventricular hypertrophy)
  8. `thalach`: Maximum heart rate achieved during exercise stress test
  9. `exang`: Exercise-induced angina (1 = yes, 0 = no)
  10. `oldpeak`: ST depression induced by exercise relative to rest
  11. `slope`: Peak exercise ST segment slope (0: upsloping, 1: flat, 2: downsloping)
  12. `ca`: Number of major vessels (0–3) colored by fluoroscopy
  13. `thal`: Thalassemia category (1: normal, 2: fixed defect, 3: reversible defect)

## c. Github Repository Link
- **GitHub Repository:** https://github.com/<your-username>/<your-repo-name>
- **Live Streamlit App:** https://<your-app-subdomain>.streamlit.app

## d. Models used:

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Logistic Regression | 0.8033 | 0.8690 | 0.7692 | 0.9091 | 0.8333 | 0.6098 |
| Decision Tree | 0.7869 | 0.8176 | 0.7500 | 0.9091 | 0.8219 | 0.5789 |
| KNN | 0.8197 | 0.8837 | 0.7895 | 0.9091 | 0.8451 | 0.6410 |
| Naive Bayes | 0.8197 | 0.8755 | 0.7895 | 0.9091 | 0.8451 | 0.6410 |
| Support Vector Machine | 0.8361 | 0.8831 | 0.7949 | 0.9394 | 0.8611 | 0.6784 |

### Observations on Model Performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Achieved a solid baseline with 80.33% accuracy and 0.8690 AUC. Its high recall (90.91%) indicates that linear boundary separation with standardized continuous inputs effectively catches patients at risk, though modest precision (76.92%) reflects some false-positive diagnoses. |
| **Decision Tree** | With pruning applied (depth limit 5), the model achieved 78.69% accuracy and 0.8176 AUC. While highly interpretable and capable of matching the 90.91% recall benchmark, the orthogonal splits produced more false positives, resulting in lower precision (75.00%) and MCC (0.5789). |
| **KNN** | Standardized Euclidean distance with $k=7$ attained 81.97% accuracy, 0.8837 AUC, and an MCC of 0.6410. Because physiological features like heart rate and ST depression naturally cluster among healthy vs. diseased patients, local neighborhood voting yields strong discriminating power. |
| **Naive Bayes** | Gaussian Naive Bayes matched KNN with 81.97% accuracy, 0.8755 AUC, and 0.6410 MCC. Despite the theoretical assumption of conditional feature independence being mildly violated by correlated physiological indicators (e.g., blood pressure and cholesterol), the probabilistic classifier maintained high diagnostic sensitivity (90.91% recall). |
| **Support Vector Machine** | Achieved the highest performance across all six evaluated metrics: 83.61% accuracy, 0.8831 AUC, 79.49% precision, 93.94% recall, 0.8611 F1 score, and 0.6784 MCC. The radial basis function (RBF) kernel effectively mapped complex non-linear clinical interactions into a higher-dimensional space where classes are more cleanly separable. |
| **Overall, Winner for your dataset?** | **Support Vector Machine (SVM with RBF Kernel)** is the undisputed overall winner. In clinical medical diagnostics, recall (sensitivity) is paramount to prevent missed diagnoses. SVM achieved both the highest Recall (0.9394) and highest Precision (0.7949), along with superior global correlation (MCC: 0.6784) and Overall Accuracy (0.8361). |

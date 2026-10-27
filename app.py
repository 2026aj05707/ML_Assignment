import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

st.set_page_config(
    page_title="Heart Disease Classification Portal",
    page_icon="❤️",
    layout="wide"
)

st.title("Heart Disease Classification & Model Evaluation")
st.markdown(
    "Interactive portal demonstrating 5 trained classification models "
    "on held-out test data without retraining."
)

# Sidebar
st.sidebar.header("1. Upload Test Data")
uploaded_file = st.sidebar.file_uploader(
    "Upload Test CSV (must contain 'target' column)",
    type=["csv"]
)

if os.path.exists("test_data.csv"):
    with open("test_data.csv", "rb") as f:
        st.sidebar.download_button(
            "Download sample test_data.csv",
            data=f,
            file_name="test_data.csv",
            mime="text/csv"
        )

st.sidebar.header("2. Model Selection")
model_options = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "KNN": "model/knn.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
    "Support Vector Machine": "model/support_vector_machine.joblib"
}

selected_model_name = st.sidebar.selectbox("Choose Classifier", list(model_options.keys()))

EXPECTED_FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]


def compute_metrics(model, X, y):
    y_pred = model.predict(X)
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X)[:, 1]
    else:
        scores = model.decision_function(X)
    return y_pred, {
        "Accuracy": accuracy_score(y, y_pred),
        "AUC": roc_auc_score(y, scores),
        "Precision": precision_score(y, y_pred, zero_division=0),
        "Recall": recall_score(y, y_pred, zero_division=0),
        "F1": f1_score(y, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y, y_pred),
    }

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        st.subheader("Uploaded Dataset Preview")
        st.dataframe(data.head(5), use_container_width=True)

        missing = [c for c in EXPECTED_FEATURES + ["target"] if c not in data.columns]
        if missing:
            st.error(f"Error: The uploaded CSV is missing required columns: {', '.join(missing)}")
        else:
            X_test = data[EXPECTED_FEATURES]
            y_test = data["target"]

            model_path = model_options[selected_model_name]
            if not os.path.exists(model_path):
                st.error(f"Model file '{model_path}' not found on server.")
            else:
                model = joblib.load(model_path)

                y_pred, metrics = compute_metrics(model, X_test, y_test)

                st.subheader(f"Evaluation Metrics: {selected_model_name}")
                col1, col2, col3 = st.columns(3)
                col4, col5, col6 = st.columns(3)

                col1.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
                col2.metric("AUC Score", f"{metrics['AUC']:.4f}")
                col3.metric("Precision", f"{metrics['Precision']:.4f}")
                col4.metric("Recall", f"{metrics['Recall']:.4f}")
                col5.metric("F1 Score", f"{metrics['F1']:.4f}")
                col6.metric("MCC Score", f"{metrics['MCC']:.4f}")

                st.markdown("---")

                # Visualizations
                v_col1, v_col2 = st.columns(2)

                with v_col1:
                    st.subheader("Confusion Matrix")
                    cm = confusion_matrix(y_test, y_pred)
                    fig, ax = plt.subplots(figsize=(4, 3))
                    sns.heatmap(
                        cm,
                        annot=True,
                        fmt="d",
                        cmap="Blues",
                        cbar=False,
                        xticklabels=["No Disease (0)", "Disease (1)"],
                        yticklabels=["No Disease (0)", "Disease (1)"],
                        ax=ax
                    )
                    ax.set_xlabel("Predicted Label")
                    ax.set_ylabel("True Label")
                    st.pyplot(fig)

                with v_col2:
                    st.subheader("Classification Report")
                    report_dict = classification_report(
                        y_test,
                        y_pred,
                        target_names=["No Disease (0)", "Disease (1)"],
                        output_dict=True
                    )
                    report_df = pd.DataFrame(report_dict).transpose()
                    st.dataframe(report_df.style.format("{:.3f}"), use_container_width=True)

                st.markdown("---")

                # Side-by-side comparison of every saved model on the same test data
                st.subheader("Comparison of All Models on Uploaded Data")
                rows = []
                for name, path in model_options.items():
                    if os.path.exists(path):
                        _, m = compute_metrics(joblib.load(path), X_test, y_test)
                        rows.append({"ML Model Name": name, **m})
                comparison_df = pd.DataFrame(rows).set_index("ML Model Name")
                st.dataframe(
                    comparison_df.style.format("{:.4f}").highlight_max(axis=0, props="background-color:#c6efce;color:black"),
                    use_container_width=True
                )
                best_model = comparison_df["MCC"].idxmax()
                st.caption(f"Best model by MCC on this data: **{best_model}**")

    except Exception as e:
        st.error(f"Error evaluating test dataset: {str(e)}")
else:
    st.info("Please upload your `test_data.csv` file from the sidebar to view evaluation results.")

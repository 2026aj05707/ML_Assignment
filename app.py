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

st.sidebar.header("2. Model Selection")
model_options = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "KNN": "model/knn.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
    "Support Vector Machine": "model/support_vector_machine.joblib"
}

selected_model_name = st.sidebar.selectbox("Choose Classifier", list(model_options.keys()))

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        st.subheader("Uploaded Dataset Preview")
        st.dataframe(data.head(5), use_container_width=True)

        if "target" not in data.columns:
            st.error("Error: The uploaded CSV must contain a 'target' column.")
        else:
            X_test = data.drop(columns=["target"])
            y_test = data["target"]

            model_path = model_options[selected_model_name]
            if not os.path.exists(model_path):
                st.error(f"Model file '{model_path}' not found on server.")
            else:
                model = joblib.load(model_path)

                # Predictions
                y_pred = model.predict(X_test)

                # Predict Proba for AUC
                if hasattr(model, "predict_proba"):
                    y_proba = model.predict_proba(X_test)[:, 1]
                    auc = roc_auc_score(y_test, y_proba)
                else:
                    auc = 0.0

                # Metrics calculation
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, zero_division=0)
                rec = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                mcc = matthews_corrcoef(y_test, y_pred)

                st.subheader(f"Evaluation Metrics: {selected_model_name}")
                col1, col2, col3 = st.columns(3)
                col4, col5, col6 = st.columns(3)

                col1.metric("Accuracy", f"{acc:.4f}")
                col2.metric("AUC Score", f"{auc:.4f}")
                col3.metric("Precision", f"{prec:.4f}")
                col4.metric("Recall", f"{rec:.4f}")
                col5.metric("F1 Score", f"{f1:.4f}")
                col6.metric("MCC Score", f"{mcc:.4f}")

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

    except Exception as e:
        st.error(f"Error evaluating test dataset: {str(e)}")
else:
    st.info("Please upload your `test_data.csv` file from the sidebar to view evaluation results.")

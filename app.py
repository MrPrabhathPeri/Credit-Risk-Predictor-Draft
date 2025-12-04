import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap

st.set_page_config(page_title="Credit Risk Predictor", layout="centered")

@st.cache_resource
def load_model():
    model = joblib.load("credit_risk_model.pkl")
    return model

model = load_model()

st.title("🏦 Credit Risk Prediction (Indian Loan Applicants)")
st.write(
    "This tool predicts the probability that a loan applicant will **default** "
    "based on their financial and demographic information."
)

st.sidebar.header("Applicant Details")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
married = st.sidebar.selectbox("Married", ["Yes", "No"])
dependents = st.sidebar.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.sidebar.selectbox("Education", ["Graduate", "Not Graduate"])
self_emp = st.sidebar.selectbox("Self Employed", ["Yes", "No"])
property_area = st.sidebar.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
credit_history = st.sidebar.selectbox("Credit History (1 = Good)", [1.0, 0.0])

applicant_income = st.sidebar.number_input("Applicant Income", min_value=0, value=5000, step=100)
coapplicant_income = st.sidebar.number_input("Coapplicant Income", min_value=0, value=0, step=100)
loan_amount = st.sidebar.number_input("Loan Amount (in thousands)", min_value=0, value=100, step=10)
loan_term = st.sidebar.number_input("Loan Amount Term (in days)", min_value=0, value=360, step=12)

input_dict = {
    "Gender": gender,
    "Married": married,
    "Dependents": dependents,
    "Education": education,
    "Self_Employed": self_emp,
    "Property_Area": property_area,
    "Credit_History": credit_history,
    "ApplicantIncome": applicant_income,
    "CoapplicantIncome": coapplicant_income,
    "LoanAmount": loan_amount,
    "Loan_Amount_Term": loan_term,
}

input_df = pd.DataFrame([input_dict])

st.subheader("📥 Input Summary")
st.write(input_df)

if st.button("Predict Risk"):
    # Predict probability of default (assuming class 1 = approved / non-default; 
    # you can invert meaning depending on how you framed it)
    proba = model.predict_proba(input_df)[0, 1]

    st.subheader("📊 Risk Score")
    st.metric(
        label="Predicted probability of **loan approval**",
        value=f"{proba * 100:.2f} %"
    )
    st.caption("You can also interpret 1 - this value as approximate default risk.")

    # SHAP explanation (local)
    try:
        # Get preprocessor & tree model from pipeline
        preprocessor = model.named_steps["preprocessor"]
        tree_model = model.named_steps["model"]

        # Transform input
        x_transformed = preprocessor.transform(input_df)

        # Feature names after preprocessing
        ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
        cat_cols = ["Gender", "Married", "Dependents", "Education",
                    "Self_Employed", "Property_Area", "Credit_History"]
        num_cols = ["ApplicantIncome", "CoapplicantIncome",
                    "LoanAmount", "Loan_Amount_Term"]
        cat_feature_names = ohe.get_feature_names_out(cat_cols)
        feature_names = np.concatenate([num_cols, cat_feature_names])

        explainer = shap.TreeExplainer(tree_model)
        shap_values = explainer.shap_values(x_transformed)

        # For binary classification, pick the SHAP values for the positive class
        sv = shap_values[0] if isinstance(shap_values, list) else shap_values

        shap_df = pd.DataFrame({
            "feature": feature_names,
            "shap_value": sv[0]
        }).sort_values("shap_value", key=lambda x: abs(x), ascending=False).head(10)

        st.subheader("🧠 Top Feature Contributions (SHAP)")
        st.write(
            "Positive SHAP values push the prediction **towards approval**, "
            "negative values push it **towards rejection**."
        )

        st.bar_chart(shap_df.set_index("feature"))

    except Exception as e:
        st.warning(
            "Could not generate SHAP explanation in this environment. "
            "You can still show SHAP plots from the notebook in your report."
        )
        st.exception(e)

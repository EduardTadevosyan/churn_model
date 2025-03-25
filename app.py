import streamlit as st
import numpy as np
from joblib import load

# Load the model
model = load('churn_model.pkl')

st.title("🔮 Customer Churn Prediction App")

# ---- User Inputs ----

SeniorCitizen = st.checkbox("Senior Citizen")
Partner = st.checkbox("Has a Partner?")
Dependents = st.checkbox("Has Dependents?")
tenure = st.slider("Tenure (in months)", 0, 72, 12)

# Contract Type (convert to numeric encoding)
contract_label = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
contract_type = {
    "Month-to-month": 0,
    "One year": 1,
    "Two year": 2
}[contract_label]

# Paperless Billing
PaperlessBilling = st.checkbox("Uses Paperless Billing?")

# Payment Method (convert to numeric encoding)
payment_label = st.selectbox("Payment Method", [
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)"
])
payment_method = {
    "Electronic check": 1,
    "Mailed check": 2,
    "Bank transfer (automatic)": 3,
    "Credit card (automatic)": 4
}[payment_label]

# One-hot features
gender_Male = st.selectbox("Gender", ["Female", "Male"]) == "Male"
MultipleLines_NoPhone = st.checkbox("No Phone Service?")
MultipleLines_Yes = st.checkbox("Has Multiple Lines?")
InternetService_Fiber = st.checkbox("Uses Fiber Optic Internet?")
OnlineSecurity_Yes = st.checkbox("Has Online Security?")
OnlineBackup_Yes = st.checkbox("Has Online Backup?")
DeviceProtection_Yes = st.checkbox("Has Device Protection?")
TechSupport_Yes = st.checkbox("Has Tech Support?")
StreamingTV_Yes = st.checkbox("Has Streaming TV?")
StreamingMovies_Yes = st.checkbox("Has Streaming Movies?")

# ---- Prepare input for model ----

features = np.array([[
    int(SeniorCitizen),
    int(Partner),
    int(Dependents),
    float(tenure),
    float(contract_type),
    int(PaperlessBilling),
    float(payment_method),
    int(gender_Male),
    int(MultipleLines_NoPhone),
    int(MultipleLines_Yes),
    int(InternetService_Fiber),
    int(OnlineSecurity_Yes),
    int(OnlineBackup_Yes),
    int(DeviceProtection_Yes),
    int(TechSupport_Yes),
    int(StreamingTV_Yes),
    int(StreamingMovies_Yes)
]], dtype=float)

# ---- Prediction ----

if st.button("Predict Churn"):
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]

    st.subheader("🔍 Prediction Result:")
    st.write("💥 **Churn**" if prediction else "✅ **No Churn**")
    st.write(f"📊 Probability of Churn: **{probability:.2%}**")

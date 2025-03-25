import streamlit as st
import numpy as np
from joblib import load

# Load the model
model = load('churn_model.pkl')

st.title("🔮 Customer Churn Prediction App")

# Input fields matching your DataFrame
SeniorCitizen = st.checkbox("Senior Citizen")
Partner = st.checkbox("Has a Partner?")
Dependents = st.checkbox("Has Dependents?")
tenure = st.slider("Tenure (months)", 0, 72, 12)

# Encoded contract values (as in your DataFrame)
contract_type = st.selectbox("Contract Type", {
    "Month-to-month": 0,
    "One year": 1,
    "Two year": 2
})

PaperlessBilling = st.checkbox("Uses Paperless Billing?")

# Encoded payment method
payment_method = st.selectbox("Payment Method", {
    "Electronic check": 1,
    "Mailed check": 2,
    "Bank transfer (automatic)": 3,
    "Credit card (automatic)": 4
})

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

# Create feature array in correct order
features = np.array([[
    int(SeniorCitizen),
    int(Partner),
    int(Dependents),
    tenure,
    contract_type,
    int(PaperlessBilling),
    payment_method,
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
]])

if st.button("Predict"):
    prediction = model.predict(features)[0]
    prob = model.predict_proba(features)[0][1]

    st.subheader("Prediction:")
    st.write("💥 Churn" if prediction else "✅ No Churn")
    st.write(f"Probability of churn: **{prob:.2%}**")

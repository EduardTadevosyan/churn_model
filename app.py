import streamlit as st
import numpy as np
from joblib import load

# Load model
model = load('churn_model.pkl')

st.title("🔮 Churn Prediction App")

# Input form
gender = st.selectbox("Gender", ['Male', 'Female'])
senior = st.checkbox("Senior Citizen?")
tenure = st.number_input("Tenure (months)", min_value=0)
monthly = st.number_input("Monthly Charges", min_value=0.0)
total = st.number_input("Total Charges", min_value=0.0)

# Convert inputs to model format
gender_encoded = 1 if gender == 'Male' else 0
senior_encoded = 1 if senior else 0

features = np.array([[gender_encoded, senior_encoded, tenure, monthly, total]])

if st.button("Predict"):
    prediction = model.predict(features)[0]
    prob = model.predict_proba(features)[0][1]
    st.markdown(f"### 🔍 Prediction: {'Churn' if prediction else 'No Churn'}")
    st.markdown(f"Probability of churn: **{prob:.2%}**")


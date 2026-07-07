import streamlit as st
import pickle
import pandas as pd

# Load trained model
import joblib
model = joblib.load("models/churn_model.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")
model_columns = joblib.load("models/model_columns.pkl")

# Page Configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)

st.title("📊 Customer Churn Prediction")
st.write("Enter the customer details below to predict churn.")

st.markdown("---")

# Input Fields
gender = st.selectbox("Gender", ["Male", "Female"])

senior = st.selectbox(
    "Senior Citizen",
    ["No", "Yes"]
)

partner = st.selectbox(
    "Partner",
    ["No", "Yes"]
)

dependents = st.selectbox(
    "Dependents",
    ["No", "Yes"]
)

tenure = st.slider(
    "Tenure (Months)",
    0,
    72,
    12
)

monthly_charges = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    max_value=200.0,
    value=70.0
)

total_charges = st.number_input(
    "Total Charges",
    min_value=0.0,
    value=1000.0
)

# Encode Inputs
gender = 1 if gender == "Male" else 0
senior = 1 if senior == "Yes" else 0
partner = 1 if partner == "Yes" else 0
dependents = 1 if dependents == "Yes" else 0

# Create DataFrame
input_data = pd.DataFrame({
    "gender": [gender],
    "SeniorCitizen": [senior],
    "Partner": [partner],
    "Dependents": [dependents],
    "tenure": [tenure],
    "MonthlyCharges": [monthly_charges],
    "TotalCharges": [total_charges]
})

# Add missing columns
for col in model_columns:
    if col not in input_data.columns:
        input_data[col] = 0

# Arrange columns in the same order as training
input_data = input_data[model_columns]

# Prediction
if st.button("Predict"):

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("⚠️ Customer is likely to churn.")
    else:
        st.success("✅ Customer is likely to stay.")

    st.subheader("Input Summary")
    st.dataframe(input_data)
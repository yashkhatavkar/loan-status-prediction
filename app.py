import streamlit as st
import pickle
import numpy as np

# Load model and encoders
model = pickle.load(open("loan.pkl", "rb"))
label_encoders = pickle.load(open("label_encoders.pkl", "rb"))

st.title("Loan Status Prediction")

# Input fields
age = st.number_input("Age", 18, 100)
gender = st.selectbox("Gender", label_encoders["Gender"].classes_)
location = st.selectbox("Location", label_encoders["Location"].classes_)
account_type = st.selectbox("Account Type", label_encoders["AccountType"].classes_)
account_balance = st.number_input("Account Balance", 0)
loan_amount = st.number_input("Loan Amount", 0)
transaction_type = st.selectbox("Transaction Type", label_encoders["TransactionType"].classes_)
transaction_amount = st.number_input("Transaction Amount", 0)

# Submit button
if st.button("Predict Loan Approval"):
    input_data = np.array([
        age,
        label_encoders["Gender"].transform([gender])[0],
        label_encoders["Location"].transform([location])[0],
        label_encoders["AccountType"].transform([account_type])[0],
        account_balance,
        loan_amount,
        label_encoders["TransactionType"].transform([transaction_type])[0],
        transaction_amount
    ]).reshape(1, -1)

    prediction = model.predict(input_data)[0]
    st.success(f"Loan Status: {'Approved' if prediction == 1 else 'Rejected'}")
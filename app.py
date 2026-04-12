import streamlit as st
import pandas as pd
import pickle

# Load model and columns
model = pickle.load(open("model/model.pkl", "rb"))
columns = pickle.load(open("model/columns.pkl", "rb"))

st.title("Customer Churn Prediction App")

st.write("Enter customer details:")
# USER INPUTS
tenure = st.slider("Tenure (months)", 0, 72)
monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0)

gender = st.selectbox("Gender", ["Male", "Female"])
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
# PREDICT BUTTON
if st.button("Predict"):

    # Create empty dataframe with all columns
    input_data = pd.DataFrame(columns=columns)
    input_data.loc[0] = 0

    # Fill numeric values
    input_data["tenure"] = tenure
    input_data["MonthlyCharges"] = monthly_charges
    input_data["TotalCharges"] = tenure * monthly_charges

    # Fill binary values
    input_data["gender"] = 1 if gender == "Female" else 0
    input_data["PaperlessBilling"] = 1 if paperless == "Yes" else 0

    # Fill one-hot encoded columns
    if contract == "One year" and "Contract_One year" in input_data.columns:
        input_data["Contract_One year"] = 1

    if contract == "Two year" and "Contract_Two year" in input_data.columns:
        input_data["Contract_Two year"] = 1
    # PREDICTION
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.error("Customer will churn ")
    else:
        st.success("Customer will stay ")
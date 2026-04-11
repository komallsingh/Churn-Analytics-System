import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================
# LOAD MODEL AND DATA
# ==========================
model   = pickle.load(open("model/model.pkl",  "rb"))
columns = pickle.load(open("model/columns.pkl","rb"))
scaler  = pickle.load(open("model/scaler.pkl", "rb"))

df = pd.read_csv("data/churn_data.csv")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(inplace=True)

# ==========================
# APP TITLE
# ==========================
st.title("Customer Churn Prediction App")
st.write("Use the tabs below to explore data or make a prediction.")

# ==========================
# TWO PAGES VIA TABS
# ==========================
tab1, tab2 = st.tabs(["📊 Dashboard", "🎯 Make Prediction"])


# --------------------------
# TAB 1 — DASHBOARD
# --------------------------
with tab1:

    st.header("Churn Overview")

    # --- Chart 1: How many customers churned ---
    st.subheader("1. How many customers churned?")
    fig, ax = plt.subplots()
    sns.countplot(data=df, x="Churn", palette={"No": "steelblue", "Yes": "tomato"}, ax=ax)
    ax.set_xlabel("Churn")
    ax.set_ylabel("Number of Customers")
    st.pyplot(fig)
    plt.close()

    # --- Chart 2: Churn by Contract Type ---
    st.subheader("2. Which contract type churns the most?")
    fig, ax = plt.subplots()
    sns.countplot(data=df, x="Contract", hue="Churn",
                  palette={"No": "steelblue", "Yes": "tomato"}, ax=ax)
    ax.set_xlabel("Contract Type")
    ax.set_ylabel("Number of Customers")
    ax.legend(title="Churn")
    st.pyplot(fig)
    plt.close()

    # --- Chart 3: Tenure vs Churn ---
    st.subheader("3. Do long-term customers churn less?")
    fig, ax = plt.subplots()
    sns.boxplot(data=df, x="Churn", y="tenure",
                palette={"No": "steelblue", "Yes": "tomato"}, ax=ax)
    ax.set_xlabel("Churn")
    ax.set_ylabel("Tenure (months)")
    st.pyplot(fig)
    plt.close()

    # --- Chart 4: Monthly Charges vs Churn ---
    st.subheader("4. Do higher charges lead to more churn?")
    fig, ax = plt.subplots()
    sns.boxplot(data=df, x="Churn", y="MonthlyCharges",
                palette={"No": "steelblue", "Yes": "tomato"}, ax=ax)
    ax.set_xlabel("Churn")
    ax.set_ylabel("Monthly Charges ($)")
    st.pyplot(fig)
    plt.close()

    # --- Chart 5: Internet Service vs Churn ---
    st.subheader("5. Which internet service has most churn?")
    fig, ax = plt.subplots()
    sns.countplot(data=df, x="InternetService", hue="Churn",
                  palette={"No": "steelblue", "Yes": "tomato"}, ax=ax)
    ax.set_xlabel("Internet Service")
    ax.set_ylabel("Number of Customers")
    ax.legend(title="Churn")
    st.pyplot(fig)
    plt.close()

    # --- Chart 6: Feature Importance ---
    st.subheader("6. Which features matter most for prediction?")
    feat_df = pd.DataFrame({
        "Feature":    columns,
        "Importance": model.feature_importances_
    })
    feat_df = feat_df.sort_values("Importance", ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(data=feat_df, x="Importance", y="Feature", palette="viridis", ax=ax)
    ax.set_xlabel("Importance Score")
    ax.set_title("Top 10 Most Important Features")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# --------------------------
# TAB 2 — PREDICTION
# --------------------------
with tab2:

    st.header("Predict Customer Churn")
    st.write("Fill in the customer details and click Predict.")

    # --- INPUT FIELDS ---
    tenure_val    = st.slider("Tenure (months)", 0, 72, 12)
    monthly_val   = st.number_input("Monthly Charges ($)", 0.0, 200.0, 65.0)
    contract_val  = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    internet_val  = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    gender_val    = st.selectbox("Gender", ["Male", "Female"])
    partner_val   = st.selectbox("Has Partner?", ["Yes", "No"])
    paperless_val = st.selectbox("Paperless Billing?", ["Yes", "No"])
    payment_val   = st.selectbox("Payment Method", [
                        "Electronic check", "Mailed check",
                        "Bank transfer (automatic)", "Credit card (automatic)"])
    security_val  = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    techsup_val   = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])

    # --- PREDICT BUTTON ---
    if st.button("Predict"):

        # Step 1: Start with all zeros
        input_data = pd.DataFrame([np.zeros(len(columns))], columns=columns)

        # Step 2: Scale the numeric columns using saved scaler
        raw = pd.DataFrame([[tenure_val, monthly_val, tenure_val * monthly_val]],
                           columns=["tenure", "MonthlyCharges", "TotalCharges"])
        scaled = scaler.transform(raw)
        input_data["tenure"]         = scaled[0][0]
        input_data["MonthlyCharges"] = scaled[0][1]
        input_data["TotalCharges"]   = scaled[0][2]

        # Step 3: Fill binary columns
        input_data["gender"]           = 1 if gender_val    == "Female" else 0
        input_data["Partner"]          = 1 if partner_val   == "Yes"    else 0
        input_data["PaperlessBilling"] = 1 if paperless_val == "Yes"    else 0

        # Step 4: Fill one-hot columns (check if column exists first)
        def fill(col, value):
            if col in columns:
                input_data[col] = value

        fill("Contract_One year",                     1 if contract_val == "One year"                   else 0)
        fill("Contract_Two year",                     1 if contract_val == "Two year"                   else 0)
        fill("InternetService_Fiber optic",           1 if internet_val == "Fiber optic"                else 0)
        fill("InternetService_No",                    1 if internet_val == "No"                         else 0)
        fill("PaymentMethod_Electronic check",        1 if payment_val  == "Electronic check"           else 0)
        fill("PaymentMethod_Mailed check",            1 if payment_val  == "Mailed check"               else 0)
        fill("PaymentMethod_Credit card (automatic)", 1 if payment_val  == "Credit card (automatic)"    else 0)
        fill("OnlineSecurity_Yes",                    1 if security_val == "Yes"                        else 0)
        fill("OnlineSecurity_No internet service",    1 if security_val == "No internet service"        else 0)
        fill("TechSupport_Yes",                       1 if techsup_val  == "Yes"                        else 0)
        fill("TechSupport_No internet service",       1 if techsup_val  == "No internet service"        else 0)

        # Step 5: Make prediction
        prediction  = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1] * 100

        # Step 6: Show result
        st.markdown("---")
        if prediction == 1:
            st.error(f"⚠️ This customer is likely to CHURN  |  Churn Probability: {probability:.1f}%")
        else:
            st.success(f"✅ This customer is likely to STAY  |  Churn Probability: {probability:.1f}%")

        # Step 7: Show simple probability bar chart
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.barplot(x=["Will Stay", "Will Churn"],
                    y=[100 - probability, probability],
                    palette=["steelblue", "tomato"], ax=ax)
        ax.set_ylabel("Probability (%)")
        ax.set_ylim(0, 100)
        ax.set_title("Prediction Confidence")
        st.pyplot(fig)
        plt.close()
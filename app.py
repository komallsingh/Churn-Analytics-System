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
# PRE-CALCULATE KEY NUMBERS
# ==========================
total_customers  = len(df)
total_churned    = (df["Churn"] == "Yes").sum()
total_stayed     = (df["Churn"] == "No").sum()
churn_rate       = total_churned / total_customers * 100

# Tenure insights
avg_tenure_churn = df[df["Churn"] == "Yes"]["tenure"].mean()
avg_tenure_stay  = df[df["Churn"] == "No"]["tenure"].mean()

# Monthly charges insights
avg_charges_churn = df[df["Churn"] == "Yes"]["MonthlyCharges"].mean()
avg_charges_stay  = df[df["Churn"] == "No"]["MonthlyCharges"].mean()

# Contract insights
contract_churn = df[df["Churn"] == "Yes"]["Contract"].value_counts()
most_churn_contract = contract_churn.idxmax()
most_churn_contract_pct = (
    df[(df["Churn"] == "Yes") & (df["Contract"] == most_churn_contract)].shape[0]
    / total_churned * 100
)

# Internet insights
internet_churn_rate = df.groupby("InternetService")["Churn"].apply(
    lambda x: (x == "Yes").mean() * 100
)
most_churn_internet = internet_churn_rate.idxmax()
most_churn_internet_pct = internet_churn_rate.max()

# Feature importance
feat_df = pd.DataFrame({
    "Feature":    columns,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=False)
top_feature = feat_df.iloc[0]["Feature"]
top_feature_score = feat_df.iloc[0]["Importance"]

# ==========================
# APP TITLE
# ==========================
st.title("📡 Customer Churn Prediction App")
st.write("Explore the data insights or go to the Predict tab to predict churn for a customer.")
st.markdown("---")

# ==========================
# TABS
# ==========================
tab1, tab2 = st.tabs(["📊 Dashboard", "🎯 Make Prediction"])


# --------------------------
# TAB 1 — DASHBOARD
# --------------------------
with tab1:

    st.header("Customer Churn Overview")

    # ── TOP KPI METRICS ──
    st.subheader("📌 Key Numbers at a Glance")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Customers",  f"{total_customers:,}")
    k2.metric("Churned",          f"{total_churned:,}")
    k3.metric("Retained",         f"{total_stayed:,}")
    k4.metric("Churn Rate",       f"{churn_rate:.1f}%")

    st.markdown("---")

    # ── ROW 1: Chart 1 + Chart 2 ──
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("1. Churned vs Stayed")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.countplot(data=df, x="Churn",
                      palette={"No": "steelblue", "Yes": "tomato"}, ax=ax)
        ax.set_xlabel("Churn")
        ax.set_ylabel("Number of Customers")
        st.pyplot(fig)
        plt.close()
        # Insight
        st.info(f"📌 Out of **{total_customers:,}** customers, "
                f"**{total_churned:,} ({churn_rate:.1f}%)** have churned. "
                f"That means roughly **1 in every {int(100/churn_rate)} customers** leaves.")

    with col2:
        st.subheader("2. Churn by Contract Type")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.countplot(data=df, x="Contract", hue="Churn",
                      palette={"No": "steelblue", "Yes": "tomato"}, ax=ax)
        ax.set_xlabel("Contract Type")
        ax.set_ylabel("Number of Customers")
        ax.legend(title="Churn")
        st.pyplot(fig)
        plt.close()
        # Insight
        st.warning(f"⚠️ **{most_churn_contract}** contracts account for "
                   f"**{most_churn_contract_pct:.1f}%** of all churned customers. "
                   f"Customers on short contracts are far more likely to leave.")

    st.markdown("---")

    # ── ROW 2: Chart 3 + Chart 4 ──
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("3. Tenure vs Churn")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.boxplot(data=df, x="Churn", y="tenure",
                    palette={"No": "steelblue", "Yes": "tomato"}, ax=ax)
        ax.set_xlabel("Churn")
        ax.set_ylabel("Tenure (months)")
        st.pyplot(fig)
        plt.close()
        # Insight
        st.success(f"✅ Customers who **stayed** had an average tenure of "
                   f"**{avg_tenure_stay:.0f} months**, while those who "
                   f"**churned** averaged only **{avg_tenure_churn:.0f} months**. "
                   f"Newer customers are at much higher risk.")

    with col4:
        st.subheader("4. Monthly Charges vs Churn")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.boxplot(data=df, x="Churn", y="MonthlyCharges",
                    palette={"No": "steelblue", "Yes": "tomato"}, ax=ax)
        ax.set_xlabel("Churn")
        ax.set_ylabel("Monthly Charges ($)")
        st.pyplot(fig)
        plt.close()
        # Insight
        st.warning(f"⚠️ Churned customers paid an average of "
                   f"**${avg_charges_churn:.2f}/month**, compared to "
                   f"**${avg_charges_stay:.2f}/month** for retained customers. "
                   f"Higher bills increase churn risk.")

    st.markdown("---")

    # ── ROW 3: Chart 5 + Chart 6 ──
    col5, col6 = st.columns(2)

    with col5:
        st.subheader("5. Churn by Internet Service")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.countplot(data=df, x="InternetService", hue="Churn",
                      palette={"No": "steelblue", "Yes": "tomato"}, ax=ax)
        ax.set_xlabel("Internet Service")
        ax.set_ylabel("Number of Customers")
        ax.legend(title="Churn")
        st.pyplot(fig)
        plt.close()
        # Insight
        st.warning(f"⚠️ **{most_churn_internet}** internet users have the highest "
                   f"churn rate at **{most_churn_internet_pct:.1f}%**. "
                   f"This may be due to higher costs or competition.")

    with col6:
        st.subheader("6. Top 10 Important Features")
        top10 = feat_df.head(10)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.barplot(data=top10, x="Importance", y="Feature",
                    palette="viridis", ax=ax)
        ax.set_xlabel("Importance Score")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        # Insight
        st.info(f"📌 The most important feature for predicting churn is "
                f"**{top_feature}** with an importance score of "
                f"**{top_feature_score:.3f}**. "
                f"Focus on this feature when identifying at-risk customers.")

    st.markdown("---")

    # ── SUMMARY BOX AT BOTTOM ──
    st.subheader("🔑 Key Takeaways")
    st.error( "🔴 **Highest Risk:**  Month-to-month contract  +  Fiber optic internet  +  Electronic check payment")
    st.warning("🟡 **Medium Risk:**  New customers (low tenure)  +  High monthly charges  +  No online security")
    st.success("🟢 **Low Risk:**     Two-year contract  +  Long tenure  +  Tech support subscribed")


# --------------------------
# TAB 2 — PREDICTION
# --------------------------
with tab2:

    st.header("🎯 Predict Customer Churn")
    st.write("Fill in the customer details below and click **Predict** to see the result.")

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Account Details")
        tenure_val    = st.slider("Tenure (months)", 0, 72, 12)
        monthly_val   = st.number_input("Monthly Charges ($)", 0.0, 200.0, 65.0)
        contract_val  = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        internet_val  = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        payment_val   = st.selectbox("Payment Method", [
                            "Electronic check", "Mailed check",
                            "Bank transfer (automatic)", "Credit card (automatic)"])

    with col_b:
        st.subheader("Personal & Service Details")
        gender_val    = st.selectbox("Gender", ["Male", "Female"])
        partner_val   = st.selectbox("Has Partner?", ["Yes", "No"])
        paperless_val = st.selectbox("Paperless Billing?", ["Yes", "No"])
        security_val  = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        techsup_val   = st.selectbox("Tech Support",    ["Yes", "No", "No internet service"])

    st.markdown("---")

    if st.button("🎯 Predict Churn"):

        # Step 1: all zeros
        input_data = pd.DataFrame([np.zeros(len(columns))], columns=columns)

        # Step 2: scale numeric
        raw    = pd.DataFrame([[tenure_val, monthly_val, tenure_val * monthly_val]],
                              columns=["tenure", "MonthlyCharges", "TotalCharges"])
        scaled = scaler.transform(raw)
        input_data["tenure"]         = scaled[0][0]
        input_data["MonthlyCharges"] = scaled[0][1]
        input_data["TotalCharges"]   = scaled[0][2]

        # Step 3: binary columns
        input_data["gender"]           = 1 if gender_val    == "Female" else 0
        input_data["Partner"]          = 1 if partner_val   == "Yes"    else 0
        input_data["PaperlessBilling"] = 1 if paperless_val == "Yes"    else 0

        # Step 4: one-hot columns
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

        # Step 5: predict
        prediction  = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1] * 100

        # Step 6: result + chart side by side
        res1, res2 = st.columns(2)

        with res1:
            st.subheader("Result")
            if prediction == 1:
                st.error( f"⚠️ This customer is likely to **CHURN**")
                st.metric("Churn Probability", f"{probability:.1f}%")
                st.warning("💡 Tip: Offer a discount or upgrade to a longer contract "
                           "to retain this customer.")
            else:
                st.success(f"✅ This customer is likely to **STAY**")
                st.metric("Churn Probability", f"{probability:.1f}%")
                st.info("💡 This customer looks stable. Keep engagement consistent.")

        with res2:
            st.subheader("Prediction Confidence")
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.barplot(x=["Will Stay", "Will Churn"],
                        y=[100 - probability, probability],
                        palette=["steelblue", "tomato"], ax=ax)
            ax.set_ylabel("Probability (%)")
            ax.set_ylim(0, 100)
            for i, v in enumerate([100 - probability, probability]):
                ax.text(i, v + 1, f"{v:.1f}%", ha="center", fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
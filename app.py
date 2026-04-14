import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

st.set_page_config(page_title="Churn Dashboard", layout="wide")

st.title("📡 Customer Churn Prediction Dashboard")
st.caption("End-to-end ML project: EDA → Model → Insights → Prediction")
st.markdown("---")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH    = os.path.join(BASE_DIR, "data",  "churn_data.csv")
MODEL_PATH   = os.path.join(BASE_DIR, "model", "model.pkl")
COLUMNS_PATH = os.path.join(BASE_DIR, "model", "columns.pkl")
SCALER_PATH  = os.path.join(BASE_DIR, "model", "scaler.pkl")
EDA_CHURN    = os.path.join(BASE_DIR, "EDA",   "eda_churn_count.png")
EDA_CONTRACT = os.path.join(BASE_DIR, "EDA",   "eda_contract.png")
EDA_TENURE   = os.path.join(BASE_DIR, "EDA",   "eda_tenure.png")
EDA_MONTHLY  = os.path.join(BASE_DIR, "EDA",   "eda_monthly_charges.png")
EDA_CORR     = os.path.join(BASE_DIR, "EDA",   "eda_correlation.png")
EDA_SUPPORT  = os.path.join(BASE_DIR, "EDA",   "eda_tech_support.png")

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return df.dropna()

df = load_data()

@st.cache_resource
def load_model():
    model   = pickle.load(open(MODEL_PATH,   "rb"))
    columns = pickle.load(open(COLUMNS_PATH, "rb"))
    scaler  = pickle.load(open(SCALER_PATH,  "rb"))
    return model, columns, scaler

model, columns, scaler = load_model()

try:
    importance = model.feature_importances_
    feat_imp = pd.DataFrame({"Feature": columns, "Importance": importance})
except:
    feat_imp = pd.DataFrame({"Feature": columns, "Importance": abs(model.coef_[0])})
feat_imp = feat_imp.sort_values("Importance", ascending=False)
top_features = feat_imp.head(5)["Feature"].tolist()

churn_df = df[df["Churn"] == "Yes"]
stay_df  = df[df["Churn"] == "No"]

total_customers = len(df)
total_churned   = len(churn_df)
total_stayed    = len(stay_df)
churn_rate      = total_churned / total_customers * 100

tab1, tab2 = st.tabs(["📊 Dashboard", "🎯 Prediction"])

with tab1:
    st.header("Business Insights Dashboard")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Customers",  f"{total_customers:,}")
    k2.metric("Churned",    f"{total_churned:,}")
    k3.metric("Retained",   f"{total_stayed:,}")
    k4.metric("Churn Rate", f"{churn_rate:.1f}%")

    st.markdown("---")

    st.subheader("1️⃣ Churn & Contract Analysis")
    col1, col2 = st.columns(2)
    col1.image(EDA_CHURN,    use_container_width=True)
    col2.image(EDA_CONTRACT, use_container_width=True)
    st.info("""
 Insight:
- Month-to-month customers show the highest churn
- Long-term contracts improve retention
 Recommendation: Encourage long-term plans with discounts
""")

    st.markdown("---")

    st.subheader("2️⃣ Tenure & Charges")
    col1, col2 = st.columns(2)
    col1.image(EDA_TENURE,  use_container_width=True)
    col2.image(EDA_MONTHLY, use_container_width=True)
    st.info("""
 Insight:
- Low tenure customers churn more
- High monthly charges increase churn risk
Recommendation: Focus on early retention offers
""")

    st.markdown("---")

    st.subheader("3️⃣ Correlation & Support")
    col1, col2 = st.columns(2)
    col1.image(EDA_CORR,    use_container_width=True)
    col2.image(EDA_SUPPORT, use_container_width=True)
    st.info("""
Insight:
- Tenure negatively correlates with churn
- Lack of tech support increases churn
 Recommendation: Improve support services
""")

with tab2:
    st.header("🎯 Predict Customer Churn")

    col1, col2 = st.columns(2)
    with col1:
        tenure   = st.slider("Tenure", 0, 72, 12)
        monthly  = st.number_input("Monthly Charges", 0.0, 200.0, 65.0)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    with col2:
        internet = st.selectbox("Internet", ["DSL", "Fiber optic", "No"])
        gender   = st.selectbox("Gender", ["Male", "Female"])

    if st.button("Predict"):

        
        input_data = pd.DataFrame([np.zeros(len(columns))], columns=columns)

        total_charges = tenure * monthly
        num_scaled = scaler.transform(
            pd.DataFrame([[tenure, monthly, total_charges]],
                         columns=["tenure", "MonthlyCharges", "TotalCharges"])
        )
        if "tenure"         in columns: input_data["tenure"]         = num_scaled[0][0]
        if "MonthlyCharges" in columns: input_data["MonthlyCharges"] = num_scaled[0][1]
        if "TotalCharges"   in columns: input_data["TotalCharges"]   = num_scaled[0][2]

        
        if "gender" in columns:
            input_data["gender"] = 1 if gender == "Female" else 0

        
        if contract == "One year"  and "Contract_One year"  in columns:
            input_data["Contract_One year"]  = 1
        elif contract == "Two year" and "Contract_Two year" in columns:
            input_data["Contract_Two year"] = 1

        
        if internet == "Fiber optic" and "InternetService_Fiber optic" in columns:
            input_data["InternetService_Fiber optic"] = 1
        elif internet == "No"         and "InternetService_No"          in columns:
            input_data["InternetService_No"]          = 1

        # ── Step 6: predict with tuned threshold (matches train.py 0.35) 
        THRESHOLD = 0.35
        prob     = model.predict_proba(input_data)[0][1]
        pred     = int(prob >= THRESHOLD)
        prob_pct = prob * 100

        reason = []
        if tenure < 12:
            reason.append("Low tenure (new customer)")
        if monthly > 80:
            reason.append("High monthly charges")
        if contract == "Month-to-month":
            reason.append("Flexible contract (higher churn risk)")
        if internet == "Fiber optic":
            reason.append("Fiber users tend to churn more")

        if pred == 1:
            st.error(f"⚠️ Likely to CHURN ({prob_pct:.1f}%)")
            if reason:
                st.warning("Possible Reasons:")
                for r in reason: st.write(f"• {r}")
        else:
            st.success(f"Likely to STAY ({prob_pct:.1f}%)")
            if reason:
                st.info("Risk Factors to Monitor:")
                for r in reason: st.write(f"• {r}")
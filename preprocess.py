import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(path):
    return pd.read_csv(path)

def clean_data(df):
    df = df.copy()
    if "customerID" in df.columns:
        df.drop(columns=["customerID"], inplace=True)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)
    return df

def encode_target(df):
    df = df.copy()
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    return df

def encode_features(df):
    df = df.copy()
    binary_cols = ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map({"Yes": 1, "No": 0, "Female": 1, "Male": 0})
    df = pd.get_dummies(df, drop_first=True)
    return df

def scale_features(df, scaler=None):
    df = df.copy()
    num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    if scaler is None:
        scaler = StandardScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])
    else:
        df[num_cols] = scaler.transform(df[num_cols])
    return df, scaler

def preprocess(path):
    df = load_data(path)
    df = clean_data(df)
    df = encode_target(df)
    df = encode_features(df)
    df, scaler = scale_features(df)  # scale at the end
    return df, scaler
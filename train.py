import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def load_and_prepare(data_path):
    df = pd.read_csv(data_path)

    # clean
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"])

    # target
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # keep only the columns our UI uses
    keep = ["tenure", "MonthlyCharges", "TotalCharges",
            "gender", "Contract", "InternetService", "Churn"]
    df = df[keep].copy()

    # binary encode gender
    df["gender"] = df["gender"].map({"Female": 1, "Male": 0})

    # one-hot encode Contract & InternetService (drop_first=True matches app.py)
    df = pd.get_dummies(df, columns=["Contract", "InternetService"], drop_first=True)

    return df


def train_model(data_path):
    df = load_and_prepare(data_path)

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    print("Features used:", X.columns.tolist())

    # scale numerics
    scaler = StandardScaler()
    num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    X[num_cols] = scaler.fit_transform(X[num_cols])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=12,
            class_weight="balanced", random_state=42
        ),
        "LogisticRegression": LogisticRegression(
            max_iter=1000, class_weight="balanced"
        ),
    }

    best_model, best_score = None, 0
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("\nK-Fold Cross Validation (scoring: recall):")
    for name, m in models.items():
        scores = cross_val_score(m, X_train, y_train, cv=skf, scoring="recall")
        print(f"  {name}: {scores.mean():.4f}")
        if scores.mean() > best_score:
            best_score = scores.mean()
            best_model = m

    print(f"\nBest model: {best_model.__class__.__name__}")
    best_model.fit(X_train, y_train)

    print("\n=== Default threshold (0.5) ===")
    y_pred = best_model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    print("\n=== Tuned threshold (0.35) ===")
    y_prob = best_model.predict_proba(X_test)[:, 1]
    y_pred2 = (y_prob >= 0.35).astype(int)
    print("Accuracy:", accuracy_score(y_test, y_pred2))
    print(classification_report(y_test, y_pred2))
    print(confusion_matrix(y_test, y_pred2))

    # save
    os.makedirs("model", exist_ok=True)
    pickle.dump(best_model,          open("model/model.pkl",   "wb"))
    pickle.dump(X.columns.tolist(),  open("model/columns.pkl", "wb"))
    pickle.dump(scaler,              open("model/scaler.pkl",  "wb"))
    print("\nSaved: model.pkl, columns.pkl, scaler.pkl")


if __name__ == "__main__":
    train_model("data/churn_data.csv")
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from preprocess import preprocess

def train_model(data_path):
    df, scaler = preprocess(data_path)   # get scaler 

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    os.makedirs("model", exist_ok=True)

    with open("model/model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open("model/columns.pkl", "wb") as f:
        pickle.dump(X.columns.tolist(), f)

    with open("model/scaler.pkl", "wb") as f:   # save scaler
        pickle.dump(scaler, f)

    print("Model, columns, and scaler saved!")

if __name__ == "__main__":
    train_model("data/churn_data.csv")
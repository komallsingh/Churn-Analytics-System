import os
import pickle
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from preprocess import preprocess


def train_model(data_path):
    
    df, scaler = preprocess(data_path)

    # 4. FEATURE SELECTION
    
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    print("\nSelected Features: ", X.columns.tolist())

  
    # 5. DATA SPLIT
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y   # 🔥 keeps class balance
    )

    # 6. MODEL SELECTION
  
    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            class_weight="balanced",
            random_state=42
        ),
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        )
    }

    best_model = None
    best_score = 0

    
    # 8. K-FOLD VALIDATION (RECALL)
    
    print("\nK-Fold Cross Validation (Optimized for Recall):")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='recall')
        avg_score = scores.mean()

        print(f"{name} Recall: {avg_score:.4f}")

        if avg_score > best_score:
            best_score = avg_score
            best_model = model

    print(f"\nBest Model Selected: {best_model}")

    
    best_model.fit(X_train, y_train)

   
    y_pred = best_model.predict(X_test)

    print("\n=== Default Threshold (0.5) ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    print("\n=== After Threshold Tuning (0.35) ===")

    y_prob = best_model.predict_proba(X_test)[:, 1]

    threshold = 0.35
    y_pred_new = (y_prob >= threshold).astype(int)

    print("Accuracy:", accuracy_score(y_test, y_pred_new))
    print("\nClassification Report:\n", classification_report(y_test, y_pred_new))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_new))

    os.makedirs("model", exist_ok=True)

    with open("model/model.pkl", "wb") as f:
        pickle.dump(best_model, f)

    with open("model/columns.pkl", "wb") as f:
        pickle.dump(X.columns.tolist(), f)

    with open("model/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    print("\nModel, columns, and scaler saved!")


if __name__ == "__main__":
    train_model("data/churn_data.csv")
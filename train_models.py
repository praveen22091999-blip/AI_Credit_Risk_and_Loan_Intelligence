"""
Day 2: Train credit risk models
Pulls merged data from MySQL, trains Logistic Regression / Random Forest / XGBoost,
picks the best model, and saves the model + SHAP explainer for use in Streamlit.
"""

import pandas as pd
import mysql.connector
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
import shap

# ---------- CONFIG: Aiven cloud MySQL (one-time migration) ----------
DB_CONFIG = {
    "host": "mysql-ab4286a-praveen22091999-9a9e.a.aivencloud.com",
    "port": 13243,
    "user": "avnadmin",
    "password": "AVNS_KjCT1GeBga-rU9jw-ow",
    "database": "defaultdb"
}
# -------------------------------------------


def load_merged_data():
    """Pull customers + credit_history + loans joined together from MySQL."""
    conn = mysql.connector.connect(**DB_CONFIG)

    query = """
        SELECT
            c.customer_id, c.age, c.income, c.employment_type, c.dependents,
            ch.credit_score, ch.credit_utilization, ch.debt_ratio,
            ch.open_credit_lines, ch.past_due_30_59, ch.past_due_60_89,
            ch.past_due_90plus, ch.real_estate_loans,
            l.loan_amount, l.tenure_months, l.interest_rate, l.loan_type,
            l.default_flag
        FROM customers c
        JOIN credit_history ch ON c.customer_id = ch.customer_id
        JOIN loans l ON c.customer_id = l.customer_id
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df


def main():
    print("Step 1: Loading merged data from MySQL...")
    df = load_merged_data()
    print(f"  Loaded {len(df)} rows.\n")

    # ---------- Feature engineering ----------
    # One-hot encode categorical columns
    df_encoded = pd.get_dummies(df, columns=["employment_type", "loan_type"], drop_first=True)

    feature_cols = [c for c in df_encoded.columns if c not in ["customer_id", "default_flag"]]
    X = df_encoded[feature_cols]
    y = df_encoded["default_flag"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Step 2: Split into {len(X_train)} train / {len(X_test)} test rows.\n")

    # ---------- Train models ----------
    print("Step 3: Training models...")
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "XGBoost": XGBClassifier(eval_metric="logloss", random_state=42)
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]

        results[name] = {
            "model": model,
            "accuracy": accuracy_score(y_test, preds),
            "roc_auc": roc_auc_score(y_test, probs),
            "precision": precision_score(y_test, preds, zero_division=0),
            "recall": recall_score(y_test, preds, zero_division=0),
            "f1": f1_score(y_test, preds, zero_division=0),
        }
        r = results[name]
        print(f"  {name:20s} | Acc={r['accuracy']:.3f}  ROC-AUC={r['roc_auc']:.3f}  "
              f"Precision={r['precision']:.3f}  Recall={r['recall']:.3f}  F1={r['f1']:.3f}")

    # ---------- Pick best model by ROC-AUC ----------
    best_name = max(results, key=lambda k: results[k]["roc_auc"])
    best_model = results[best_name]["model"]
    print(f"\nStep 4: Best model = {best_name} (ROC-AUC={results[best_name]['roc_auc']:.3f})")

    # ---------- Save model + feature list ----------
    joblib.dump(best_model, "credit_risk_model.pkl")
    joblib.dump(feature_cols, "model_features.pkl")
    print("  Saved credit_risk_model.pkl and model_features.pkl")

    # ---------- SHAP explainer ----------
    print("\nStep 5: Building SHAP explainer...")
    if best_name in ["Random Forest", "XGBoost"]:
        explainer = shap.TreeExplainer(best_model)
    else:
        explainer = shap.LinearExplainer(best_model, X_train)

    shap_values = explainer.shap_values(X_test)
    joblib.dump(explainer, "shap_explainer.pkl")
    print("  Saved shap_explainer.pkl")

    # Save a summary plot as a sanity check
    plt.figure()
    if isinstance(shap_values, list):  # some explainers return a list per class
        shap.summary_plot(shap_values[1], X_test, show=False)
    else:
        shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig("shap_summary.png", dpi=120)
    print("  Saved shap_summary.png (open this to sanity-check feature importance)")

    # ---------- Save comparison table for resume/portfolio ----------
    comparison_df = pd.DataFrame(results).T[["accuracy", "roc_auc", "precision", "recall", "f1"]]
    comparison_df.to_csv("model_comparison.csv")
    print("\nModel comparison saved to model_comparison.csv:")
    print(comparison_df.round(3))

    print("\nDay 2 complete. Files ready for Streamlit: "
          "credit_risk_model.pkl, model_features.pkl, shap_explainer.pkl")


if __name__ == "__main__":
    main()

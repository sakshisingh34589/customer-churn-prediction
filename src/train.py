"""
train.py
--------
Trains Logistic Regression and Random Forest classifiers on the Telco Churn
dataset, compares their performance, and saves the best full pipeline.

Usage (from customer-churn-prediction/ directory):
    python src/train.py
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
)

# Make src/ importable when running from project root
sys.path.insert(0, os.path.dirname(__file__))
from preprocess import load_and_clean, build_pipeline, split_data

# ── Output path ──────────────────────────────────────────────────────────────
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)
MODEL_SAVE_PATH = os.path.join(MODELS_DIR, "best_model_pipeline.pkl")


def evaluate_model(pipeline, X_test, y_test, model_name: str) -> dict:
    """
    Generate all evaluation metrics for a fitted pipeline.
    Returns a dict of metric name → value.
    """
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "Model":     model_name,
        "Accuracy":  round(accuracy_score(y_test, y_pred),  4),
        "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall":    round(recall_score(y_test, y_pred,    zero_division=0), 4),
        "F1-Score":  round(f1_score(y_test, y_pred,        zero_division=0), 4),
        "ROC-AUC":   round(roc_auc_score(y_test, y_prob),  4),
    }
    return metrics


def main():
    print("=" * 60)
    print("  Customer Churn Prediction — Model Training")
    print("=" * 60)

    # ── 1. Load and clean data ────────────────────────────────────────────────
    df = load_and_clean()

    # ── 2. Split data ─────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = split_data(df)

    # ── 3. Define models ──────────────────────────────────────────────────────
    # class_weight='balanced' helps with the ~26% churn class imbalance
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight="balanced",
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        ),
    }

    # ── 4. Train and evaluate each model ──────────────────────────────────────
    results = []
    fitted_pipelines = {}

    for model_name, model in models.items():
        print(f"\n[INFO] Training: {model_name} ...")
        pipeline = build_pipeline(model)
        pipeline.fit(X_train, y_train)
        fitted_pipelines[model_name] = pipeline

        metrics = evaluate_model(pipeline, X_test, y_test, model_name)
        results.append(metrics)

        # Print detailed classification report
        y_pred = pipeline.predict(X_test)
        print(f"\n  Classification Report — {model_name}:")
        print(classification_report(y_test, y_pred,
                                    target_names=["No Churn", "Churn"]))

    # ── 5. Comparison table ───────────────────────────────────────────────────
    results_df = pd.DataFrame(results).set_index("Model")
    print("\n" + "=" * 60)
    print("  Model Comparison")
    print("=" * 60)
    print(results_df.to_string())
    print()

    # ── 6. Select best model by ROC-AUC (tiebreak: F1-Score) ─────────────────
    # ROC-AUC is preferred over accuracy because the dataset is imbalanced (~26% churn)
    best_name = results_df["ROC-AUC"].idxmax()
    # Check for tie
    max_auc = results_df["ROC-AUC"].max()
    tied = results_df[results_df["ROC-AUC"] == max_auc]
    if len(tied) > 1:
        best_name = tied["F1-Score"].idxmax()

    best_pipeline = fitted_pipelines[best_name]
    best_metrics  = results_df.loc[best_name]

    print(f"[RESULT] Best model: {best_name}")
    print(f"         ROC-AUC  : {best_metrics['ROC-AUC']}")
    print(f"         F1-Score : {best_metrics['F1-Score']}")
    print(f"         Accuracy : {best_metrics['Accuracy']}")

    # ── 7. Save the best pipeline ─────────────────────────────────────────────
    joblib.dump(best_pipeline, MODEL_SAVE_PATH)
    print(f"\n[SAVED] Best pipeline saved to: {MODEL_SAVE_PATH}")
    print(f"        Model type: {type(best_pipeline.named_steps['classifier']).__name__}")

    # ── 8. Save results table for report ──────────────────────────────────────
    results_path = os.path.join(MODELS_DIR, "model_comparison.csv")
    results_df.to_csv(results_path)
    print(f"[SAVED] Comparison table saved to: {results_path}")

    print("\n[DONE] Training complete.\n")
    return best_name, results_df


if __name__ == "__main__":
    main()

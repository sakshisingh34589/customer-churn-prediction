"""
evaluate.py
-----------
Loads the saved best model pipeline and generates evaluation artefacts:
  - models/confusion_matrix.png
  - models/roc_curve.png

Usage (from customer-churn-prediction/ directory):
    python src/evaluate.py
"""

import os
import sys
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix, roc_curve, roc_auc_score, ConfusionMatrixDisplay
)

sys.path.insert(0, os.path.dirname(__file__))
from preprocess import load_and_clean, split_data

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH = os.path.join(MODELS_DIR, "best_model_pipeline.pkl")


def plot_confusion_matrix(y_test, y_pred, save_path: str):
    """Save a labelled confusion matrix heatmap."""
    cm = confusion_matrix(y_test, y_pred)
    labels = ["No Churn (0)", "Churn (1)"]

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels, ax=ax,
        linewidths=0.5, linecolor="gray",
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[SAVED] Confusion matrix -> {save_path}")


def plot_roc_curve(y_test, y_prob, save_path: str):
    """Save ROC curve with AUC annotated."""
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#3b82d4", lw=2,
            label=f"ROC Curve (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1,
            label="Random Baseline")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[SAVED] ROC curve -> {save_path}")


def main():
    print("=" * 60)
    print("  Customer Churn Prediction — Evaluation Plots")
    print("=" * 60)

    # Load the saved pipeline
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model not found at {MODEL_PATH}")
        print("        Run  python src/train.py  first.")
        sys.exit(1)

    pipeline = joblib.load(MODEL_PATH)
    model_name = type(pipeline.named_steps["classifier"]).__name__
    print(f"[INFO] Loaded pipeline: {model_name}")

    # Recreate the same test split (random_state=42 guarantees identical split)
    df = load_and_clean()
    _, X_test, _, y_test = split_data(df)

    # Generate predictions
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    # Plot and save
    plot_confusion_matrix(
        y_test, y_pred,
        save_path=os.path.join(MODELS_DIR, "confusion_matrix.png"),
    )
    plot_roc_curve(
        y_test, y_prob,
        save_path=os.path.join(MODELS_DIR, "roc_curve.png"),
    )

    print("\n[DONE] Evaluation complete.\n")


if __name__ == "__main__":
    main()

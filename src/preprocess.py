"""
preprocess.py
-------------
Handles data loading, cleaning, and building a leakage-safe preprocessing pipeline.

Usage (from project root):
    from src.preprocess import load_and_clean, build_pipeline, split_data
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# ── Path to the raw dataset ─────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data",
                         "WA_Fn-UseC_-Telco-Customer-Churn.csv")

# ── Column groups ────────────────────────────────────────────────────────────
# Numeric features that will be scaled
NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]

# Multi-value categorical features — will be one-hot encoded
CATEGORICAL_FEATURES = [
    "InternetService",
    "Contract",
    "PaymentMethod",
    "MultipleLines",
]

# Binary Yes/No features — will be imputed (as string) + OHE
# (OHE with drop='if_binary' turns them into a single 0/1 column)
BINARY_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "PaperlessBilling",
]

# SeniorCitizen is already 0/1 integer — treated as numeric
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES + ["SeniorCitizen"]
TARGET = "Churn"


def load_and_clean(path: str = DATA_PATH) -> pd.DataFrame:
    """
    Load the CSV and perform all cleaning steps.

    Steps:
      1. Load CSV
      2. Drop customerID (not a feature)
      3. Consolidate "No internet service" / "No phone service" → "No"
      4. Convert TotalCharges to numeric (whitespace strings become NaN)
      5. Drop the ~11 rows where TotalCharges is NaN after conversion
      6. Check and report duplicates
      7. Encode Churn as 1 (Yes) / 0 (No)

    Returns a cleaned DataFrame.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\n\n  Dataset not found at: {os.path.abspath(path)}\n"
            "  Please download the IBM Telco Customer Churn dataset and place it at:\n"
            "      customer-churn-prediction/data/WA_Fn-UseC_-Telco-Customer-Churn.csv\n"
            "  Source: https://www.kaggle.com/datasets/blastchar/telco-customer-churn\n"
        )

    df = pd.read_csv(path)
    print(f"[INFO] Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    # 1. Drop customerID — it is an identifier, not a predictive feature
    df = df.drop(columns=["customerID"])

    # 2. Consolidate redundant values in service columns
    #    "No internet service" and "No phone service" are functionally "No"
    service_cols = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies", "MultipleLines",
    ]
    for col in service_cols:
        df[col] = df[col].replace(
            {"No internet service": "No", "No phone service": "No"}
        )

    # 3. TotalCharges arrives as strings; whitespace entries → NaN
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    missing_tc = df["TotalCharges"].isna().sum()
    if missing_tc > 0:
        print(f"[INFO] Dropping {missing_tc} rows with missing TotalCharges "
              f"(these are new customers with no charges yet)")
        df = df.dropna(subset=["TotalCharges"])

    # 4. Check duplicates
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        print(f"[WARN] Found {duplicates} duplicate rows — dropping them")
        df = df.drop_duplicates()
    else:
        print("[INFO] No duplicate rows found")

    # 5. Encode target column
    df[TARGET] = df[TARGET].map({"Yes": 1, "No": 0})
    churn_rate = df[TARGET].mean() * 100
    print(f"[INFO] Churn rate: {churn_rate:.1f}%  "
          f"(No={( df[TARGET]==0).sum()}, Yes={(df[TARGET]==1).sum()})")

    return df.reset_index(drop=True)


def build_pipeline(model) -> Pipeline:
    """
    Build a full scikit-learn Pipeline: ColumnTransformer + classifier.

    The pipeline handles:
      - Numeric features: SimpleImputer (median) + StandardScaler
      - Categorical features: SimpleImputer (most_frequent) + OneHotEncoder
      - Binary features: SimpleImputer (most_frequent) + OneHotEncoder (drop if_binary)
      - SeniorCitizen (already 0/1): passthrough

    Parameters
    ----------
    model : scikit-learn estimator
        An unfitted classifier (e.g. LogisticRegression).

    Returns
    -------
    sklearn.pipeline.Pipeline
    """
    # Numeric pipeline
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    # Multi-class categorical pipeline
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")),
    ])

    # Binary (Yes/No) categorical pipeline
    binary_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(drop="if_binary", sparse_output=False, handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num",  numeric_pipe,     NUMERIC_FEATURES),
            ("cat",  categorical_pipe, CATEGORICAL_FEATURES),
            ("bin",  binary_pipe,      BINARY_FEATURES),
            ("pass", "passthrough",    ["SeniorCitizen"]),
        ],
        remainder="drop",
    )

    full_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier",   model),
    ])

    return full_pipeline


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Split DataFrame into train/test sets with stratification on target.

    Returns X_train, X_test, y_train, y_test
    """
    X = df[ALL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )
    print(f"[INFO] Train size: {len(X_train)}  |  Test size: {len(X_test)}")
    return X_train, X_test, y_train, y_test

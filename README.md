# Customer Churn Prediction using Machine Learning

> **AICTE-IBM Machine Learning Internship Project**

---

## Problem Statement

Telecom companies lose significant revenue every year when customers cancel their subscriptions — a phenomenon known as **customer churn**. Acquiring a new customer costs 5–7× more than retaining an existing one. The goal of this project is to build a machine learning model that can predict, in advance, which customers are likely to churn so that the business can take proactive retention action.

---

## Objective

Build a complete machine learning pipeline that:
1. Cleans and explores the IBM Telco Customer Churn dataset
2. Trains and compares Logistic Regression and Random Forest classifiers
3. Selects the best model based on ROC-AUC and F1-score
4. Deploys the model as an interactive Streamlit web application

---

## Dataset

| Property | Value |
|----------|-------|
| **Name** | IBM Telco Customer Churn |
| **Source** | [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) |
| **File** | `WA_Fn-UseC_-Telco-Customer-Churn.csv` |
| **Rows** | 7,043 customers |
| **Columns** | 21 features + 1 target |
| **Target** | `Churn` (Yes = 1 / No = 0) |
| **Churn Rate** | ~26.5% |

### Key Features
- **Demographics:** gender, SeniorCitizen, Partner, Dependents
- **Services:** PhoneService, InternetService, OnlineSecurity, StreamingTV, etc.
- **Account:** tenure, Contract, PaymentMethod, MonthlyCharges, TotalCharges

---

## Technologies Used

| Tool | Purpose |
|------|---------|
| Python 3.9+ | Core programming language |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Matplotlib / Seaborn | Data visualisation |
| Scikit-learn | ML pipeline, models, evaluation |
| Joblib | Model serialisation |
| Streamlit | Web application |
| Jupyter | EDA notebook |

---

## Machine Learning Workflow

```
Raw CSV
   ↓
Data Cleaning  (fix TotalCharges, drop IDs, consolidate service columns)
   ↓
EDA  (distributions, correlations, churn patterns)
   ↓
Preprocessing Pipeline  (imputer → scaler / OHE → ColumnTransformer)
   ↓
Model Training  (Logistic Regression | Random Forest)
   ↓
Evaluation  (Accuracy, Precision, Recall, F1, ROC-AUC)
   ↓
Best Model Saved  (best_model_pipeline.pkl)
   ↓
Streamlit App
```

---

## Data Preprocessing

The preprocessing is handled inside a **scikit-learn Pipeline** — this prevents data leakage (no information from the test set leaks into training).

| Step | Detail |
|------|--------|
| Drop `customerID` | Not a predictive feature |
| Fix `TotalCharges` | Convert whitespace strings → NaN → drop ~11 rows |
| Consolidate service columns | `"No internet service"` / `"No phone service"` → `"No"` |
| Encode target | `Yes` → 1, `No` → 0 |
| Numeric features | `SimpleImputer(median)` + `StandardScaler` |
| Multi-class categorical | `SimpleImputer` + `OneHotEncoder(drop='first')` |
| Binary categorical | `SimpleImputer` + `OneHotEncoder(drop='if_binary')` |
| Train/test split | 80/20, `stratify=y`, `random_state=42` |

---

## Models Used

### Logistic Regression
A linear model that estimates the probability of churn using a logistic (sigmoid) function. It is simple, fast, and easy to interpret. Good as a baseline model.

### Random Forest Classifier
An ensemble of many decision trees. Each tree votes and the majority wins. More powerful than Logistic Regression because it can capture non-linear relationships.

Both models use `class_weight='balanced'` to handle the ~26% churn class imbalance.

---

## Evaluation Metrics

| Metric | Why it matters for this problem |
|--------|----------------------------------|
| **Accuracy** | Overall correctness (misleading when classes are imbalanced) |
| **Precision** | Of predicted churners, how many actually churned? |
| **Recall** | Of actual churners, how many did we catch? (very important) |
| **F1-Score** | Harmonic mean of Precision and Recall |
| **ROC-AUC** | Overall ability to distinguish churners from non-churners (primary metric) |

> **Why ROC-AUC over Accuracy?** With ~26% churn, a naive model that always predicts "No Churn" achieves ~73% accuracy — but catches zero churners. ROC-AUC punishes this.

---

## Results

*Run `python src/train.py` to generate actual results. The table below will be populated after training.*

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.7439 | 0.5106 | 0.7763 | 0.6160 | **0.8463** ✅ |
| Random Forest | 0.7739 | 0.5659 | 0.6253 | 0.5941 | 0.8248 |

**Winner: Logistic Regression** — selected by highest ROC-AUC (0.8463).
The winning model is saved to `models/best_model_pipeline.pkl`.

---

## Project Structure

```
customer-churn-prediction/
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv   ← place dataset here
├── notebooks/
│   └── churn_analysis.ipynb                    ← EDA notebook
├── src/
│   ├── preprocess.py                           ← data loading + pipeline builder
│   ├── train.py                                ← model training + selection
│   └── evaluate.py                             ← evaluation plots
├── models/
│   ├── best_model_pipeline.pkl                 ← saved model (after training)
│   ├── model_comparison.csv                    ← metrics table
│   ├── confusion_matrix.png                    ← evaluation plot
│   └── roc_curve.png                           ← evaluation plot
├── app.py                                      ← Streamlit web application
├── requirements.txt
├── README.md
└── PROJECT_REPORT.md
```

---

## Installation

```bash
# 1. Clone or download this project
cd customer-churn-prediction

# 2. Create a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Step 1 — Place the Dataset

Download `WA_Fn-UseC_-Telco-Customer-Churn.csv` from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) and place it in:

```
customer-churn-prediction/data/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

### Step 2 — Train the Model

```bash
python src/train.py
```

This will:
- Clean and preprocess the data
- Train Logistic Regression and Random Forest
- Print a comparison table
- Save the best model to `models/best_model_pipeline.pkl`

### Step 3 — Generate Evaluation Plots (Optional)

```bash
python src/evaluate.py
```

Saves `models/confusion_matrix.png` and `models/roc_curve.png`.

### Step 4 — Run the Streamlit App

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

### Step 5 — Open the EDA Notebook (Optional)

```bash
jupyter notebook notebooks/churn_analysis.ipynb
```

---

## Example Usage

Launch the Streamlit app, then enter customer details:

| Field | Example Value |
|-------|--------------|
| Tenure | 2 months |
| Monthly Charges | $85 |
| Contract | Month-to-month |
| Internet Service | Fiber optic |
| Payment Method | Electronic check |

**Expected output:** ⚠️ Customer Likely to Churn (~80% probability)

---

## Key Findings

1. **Contract type** is the strongest predictor — month-to-month customers churn at ~43%
2. **New customers** (tenure < 12 months) are high-risk
3. **Fiber optic** users churn more than DSL (~42% vs 19%)
4. **Electronic check** payment method strongly associated with churn
5. **High monthly charges** correlate with increased churn

---

## Future Improvements

- Hyperparameter tuning using GridSearchCV / RandomizedSearchCV
- SMOTE oversampling to handle class imbalance during training
- XGBoost or LightGBM for potentially better performance
- SHAP values for explainable AI (feature importance per prediction)
- Cross-validation (k-fold) for more robust evaluation

---

## Author

AICTE-IBM Machine Learning Internship
"# customer-churn-prediction" 

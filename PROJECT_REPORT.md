# Project Report: Customer Churn Prediction using Machine Learning

**Submitted by:** AICTE-IBM Machine Learning Internship Participant  
**Internship Programme:** AICTE-IBM Skill Development Initiative  
**Project Title:** Customer Churn Prediction using Machine Learning  

---

## 1. Introduction

Customer churn — the phenomenon where customers discontinue their subscription to a service — is one of the most critical business challenges in the telecommunications industry. When a customer leaves, the company loses not just the recurring revenue, but also the investment made in acquiring and retaining that customer. Research indicates that acquiring a new customer costs five to seven times more than retaining an existing one. Therefore, the ability to identify customers at risk of churning before they actually leave is extremely valuable.

Machine learning provides a data-driven approach to this problem. By learning patterns from historical customer data — including demographics, account information, and service usage — a trained model can predict whether a given customer is likely to churn in the near future. This enables the business to intervene proactively with targeted retention strategies such as personalised discounts, contract upgrades, or dedicated support.

This project implements an end-to-end machine learning pipeline for customer churn prediction using the publicly available IBM Telco Customer Churn dataset. The project covers data cleaning, exploratory data analysis, feature engineering, model training, evaluation, and a web-based prediction interface built with Streamlit.

---

## 2. Problem Statement

Given a dataset of telecom customer records containing demographic information, account details, and service usage, build a supervised binary classification model that can accurately predict whether a customer will churn (leave the service) or not.

**Target Variable:** `Churn`  
- Class 1: Customer will churn (Yes)  
- Class 0: Customer will not churn (No)

---

## 3. Objectives

The specific objectives of this project are:

1. Load, inspect, and clean the IBM Telco Customer Churn dataset.
2. Perform Exploratory Data Analysis (EDA) to identify patterns associated with churn.
3. Build a leakage-safe preprocessing pipeline using scikit-learn.
4. Train and compare two machine learning models: Logistic Regression and Random Forest Classifier.
5. Evaluate the models using multiple performance metrics including ROC-AUC, F1-score, Precision, Recall, and Accuracy.
6. Select the best-performing model and save it for deployment.
7. Build a Streamlit web application that allows users to input customer details and receive a real-time churn prediction.

---

## 4. Dataset Description

The IBM Telco Customer Churn dataset is a widely used benchmark dataset in the machine learning community. It contains records of 7,043 customers of a fictional telecommunications company based in California.

| Property | Value |
|----------|-------|
| Total Records | 7,043 |
| Total Features | 20 (after dropping customerID) |
| Target Column | Churn (Yes / No) |
| Churn Rate | ~26.5% |
| Source | IBM Sample Data / Kaggle |

### Feature Categories

**Demographic Features:**
- `gender`: Male or Female
- `SeniorCitizen`: Whether the customer is a senior citizen (0 or 1)
- `Partner`: Whether the customer has a partner
- `Dependents`: Whether the customer has dependents

**Service Features:**
- `PhoneService`, `MultipleLines`: Phone-related services
- `InternetService`: DSL, Fiber optic, or No internet
- `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`: Add-on services
- `StreamingTV`, `StreamingMovies`: Streaming services

**Account Features:**
- `tenure`: Number of months the customer has been with the company
- `Contract`: Month-to-month, One year, or Two year
- `PaperlessBilling`, `PaymentMethod`: Billing information
- `MonthlyCharges`, `TotalCharges`: Charges information

### Known Data Quality Issue

The `TotalCharges` column is stored as a string (object dtype) rather than a number. Approximately 11 rows contain whitespace (`" "`) in this column instead of a numeric value. These rows correspond to customers with `tenure = 0` (brand new customers with no accumulated charges). These rows were converted using `pd.to_numeric(..., errors='coerce')` which converts whitespace to `NaN`, and the resulting NaN rows were dropped. This affects only 0.15% of the dataset.

---

## 5. Tools and Technologies

| Tool / Library | Version | Purpose |
|---------------|---------|---------|
| Python | 3.9+ | Core programming language |
| Pandas | ≥1.5 | Data loading and manipulation |
| NumPy | ≥1.23 | Numerical computations |
| Matplotlib | ≥3.6 | Data visualisation |
| Seaborn | ≥0.12 | Statistical visualisation |
| Scikit-learn | ≥1.2 | ML pipeline, models, evaluation |
| Joblib | ≥1.2 | Model serialisation |
| Streamlit | ≥1.28 | Web application |
| Jupyter | ≥1.0 | EDA notebook |

---

## 6. Data Preprocessing

A critical principle followed in this project is that all preprocessing steps are encapsulated inside a scikit-learn `Pipeline`. This guarantees that the same transformations applied during training are automatically applied at prediction time — preventing data leakage and ensuring consistency between training and production.

### 6.1 Cleaning Steps (before pipeline)

1. **Drop `customerID`**: This column is an identifier and has no predictive value.
2. **Consolidate service columns**: Values like `"No internet service"` and `"No phone service"` are replaced with `"No"`, since they are functionally identical to not having the service.
3. **Fix `TotalCharges`**: Convert from string to float; drop 11 rows with missing values.
4. **Encode target**: Map `Churn` column — `"Yes"` → 1, `"No"` → 0.

### 6.2 Train/Test Split

The dataset is split into 80% training and 20% testing using:
- `test_size = 0.2`
- `random_state = 42` (for reproducibility)
- `stratify = y` (ensures both splits maintain the ~26% churn ratio)

### 6.3 Pipeline Structure

The `ColumnTransformer` applies different preprocessing to each feature group:

| Feature Group | Features | Transformer |
|--------------|---------|------------|
| Numeric | tenure, MonthlyCharges, TotalCharges | SimpleImputer(median) → StandardScaler |
| Multi-class categorical | InternetService, Contract, PaymentMethod, MultipleLines | SimpleImputer(most_frequent) → OneHotEncoder(drop='first') |
| Binary categorical | gender, Partner, Dependents, + 8 service columns | SimpleImputer(most_frequent) → OneHotEncoder(drop='if_binary') |
| Already numeric | SeniorCitizen | Passthrough |

The full pipeline is: `ColumnTransformer → Classifier`

---

## 7. Exploratory Data Analysis

### 7.1 Churn Distribution

The dataset is **imbalanced** — approximately 73.5% of customers did not churn while 26.5% did. This class imbalance means that accuracy is not a reliable performance metric. A model that always predicts "No Churn" would achieve ~73% accuracy while identifying zero actual churners. This motivates the use of ROC-AUC and F1-score as primary metrics.

### 7.2 Churn by Contract Type

Month-to-month contract customers exhibit a churn rate of approximately 43%, compared to ~11% for one-year contracts and ~3% for two-year contracts. Contract type is the single strongest predictor of churn in this dataset.

### 7.3 Churn by Tenure

Short-tenured customers (1–12 months) show significantly higher churn rates compared to long-tenured customers. Customers who have been with the company for over 3 years almost never churn. This suggests that early retention strategies are critical.

### 7.4 Churn by Monthly Charges

Customers who churned had higher average monthly charges (~$74) compared to those who stayed (~$61). This may indicate price sensitivity or dissatisfaction with the perceived value for money.

### 7.5 Churn by Internet Service

Fiber optic customers churn at approximately 42%, compared to ~19% for DSL customers and ~7% for customers with no internet service. Fiber optic is a premium service with higher charges, potentially attracting price-sensitive customers.

### 7.6 Churn by Payment Method

Electronic check users have the highest churn rate (~45%), while customers using automatic payment methods (bank transfer or credit card) churn at significantly lower rates (~15–17%). Automatic payments may indicate higher commitment to the service.

---

## 8. Model Development

### 8.1 Logistic Regression

Logistic Regression is a linear classification algorithm that models the probability of a binary outcome using the logistic (sigmoid) function. It is interpretable, computationally inexpensive, and serves as an excellent baseline. Key parameters used:

- `max_iter = 1000`: Allows sufficient convergence time on this dataset
- `class_weight = 'balanced'`: Adjusts for class imbalance by weighting minority class higher
- `random_state = 42`: Reproducibility

### 8.2 Random Forest Classifier

Random Forest is an ensemble method that builds many decision trees on random subsets of the data and features, then aggregates their predictions by majority vote. It handles non-linear relationships and interactions between features effectively. Key parameters:

- `n_estimators = 100`: 100 decision trees in the forest
- `class_weight = 'balanced'`: Class imbalance correction
- `random_state = 42`: Reproducibility

### 8.3 Model Selection Criterion

The winning model is selected based on **ROC-AUC score** (primary), with F1-score as tiebreaker. ROC-AUC is preferred because it measures the model's ability to separate churners from non-churners across all probability thresholds, and is robust to class imbalance.

---

## 9. Model Evaluation

### 9.1 Evaluation Metrics Explained

- **Accuracy**: Fraction of all predictions that were correct. Misleading with imbalanced classes.
- **Precision**: Of all customers predicted to churn, what fraction actually churned? (Avoid false alarms)
- **Recall**: Of all customers who actually churned, what fraction did we catch? (Avoid missing churners — this is critical)
- **F1-Score**: Harmonic mean of Precision and Recall. A balance between the two.
- **ROC-AUC**: Area Under the ROC Curve. Measures discrimination ability across all thresholds. 1.0 is perfect; 0.5 is random.

### 9.2 Results

*Results obtained on held-out test set (20% of data = 1,402 customers):*

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.7439 | 0.5106 | 0.7763 | 0.6160 | **0.8463** ✅ |
| Random Forest | 0.7739 | 0.5659 | 0.6253 | 0.5941 | 0.8248 |

**Selected Model: Logistic Regression**
**Selection Reason:** Highest ROC-AUC score (0.8463 vs 0.8248) on the held-out test set (20% of data, 1,402 customers). Although Random Forest achieves slightly higher accuracy (0.7739 vs 0.7439), ROC-AUC is the more reliable metric for an imbalanced dataset. Logistic Regression also achieves significantly better Recall (0.7763 vs 0.6253), meaning it catches more actual churners — which is the primary business objective.

### 9.3 Confusion Matrix Interpretation

The confusion matrix shows four outcomes:
- **True Positive (TP):** Correctly predicted churn
- **True Negative (TN):** Correctly predicted no churn
- **False Positive (FP):** Predicted churn but customer stayed (false alarm)
- **False Negative (FN):** Predicted no churn but customer left (missed churner — most costly)

In a business context, **False Negatives** are more costly than False Positives because missing a churner means losing a customer with no chance to intervene.

---

## 10. Results

The final selected model is **Logistic Regression** with `class_weight='balanced'` and `max_iter=1000`, wrapped in a full scikit-learn Pipeline.

**Key Results (test set, 1,402 customers):**
- **ROC-AUC: 0.8463** — The model correctly ranks a randomly selected churner above a non-churner 84.6% of the time.
- **Recall: 0.7763** — The model correctly identifies 77.6% of all actual churners.
- **F1-Score: 0.6160** — A balanced measure of Precision and Recall for the churn class.

**Interesting Finding:** Logistic Regression outperformed Random Forest on ROC-AUC despite being a simpler model. This is consistent with the literature on this dataset — the churn pattern in this data has strong linear signals (contract type, tenure, monthly charges) that a linear model captures effectively. The class balancing (`class_weight='balanced'`) was critical for improving Recall on the minority churn class.

The complete confusion matrix and ROC curve are saved as PNG files in the `models/` directory.

---

## 11. Streamlit Application

A web-based prediction interface was built using Streamlit (`app.py`). The application allows a business user — who does not need to know Python or machine learning — to enter customer details through a simple form and receive an instant churn prediction.

### Key Features

| Feature | Description |
|---------|-------------|
| Input form | 19 customer feature inputs (sliders, dropdowns, radio buttons) |
| Prediction | Binary classification: Churn / No Churn |
| Probability | Churn probability percentage shown with a progress bar |
| Explanation | Human-readable explanation of factors influencing the prediction |
| Model loading | Uses `@st.cache_resource` to load the pipeline once efficiently |

### How It Works

1. The user fills in customer details (contract type, tenure, monthly charges, services used, etc.)
2. On clicking **"Predict Churn"**, the app constructs a one-row `pandas.DataFrame` with the input values
3. The saved pipeline (`best_model_pipeline.pkl`) is loaded — it contains both the preprocessor and the model
4. The pipeline transforms the input and generates a prediction and probability
5. The result is displayed with colour coding (red for churn risk, green for retention)

---

## 12. Conclusion

This project successfully demonstrates an end-to-end machine learning pipeline for customer churn prediction using the IBM Telco Customer Churn dataset. The following were achieved:

1. **Data was cleaned and explored** — identifying key churn drivers: contract type, tenure, monthly charges, internet service type, and payment method.
2. **A leakage-safe preprocessing pipeline** was built using scikit-learn's `ColumnTransformer` and `Pipeline`, ensuring that test data was never used to fit transformations.
3. **Two models were trained and compared** — Logistic Regression and Random Forest — using five evaluation metrics.
4. **The best model was selected objectively** based on ROC-AUC score and saved using Joblib.
5. **A Streamlit web application** was deployed locally, making the model accessible to non-technical business users.

The project demonstrates core machine learning competencies including data preprocessing, feature engineering, model evaluation, and model deployment — all relevant to real-world data science roles.

---

## 13. Future Scope

1. **Hyperparameter tuning**: Use `GridSearchCV` or `RandomizedSearchCV` to systematically find optimal model parameters.
2. **SMOTE oversampling**: Apply Synthetic Minority Over-sampling Technique to address class imbalance during training.
3. **Additional models**: Experiment with XGBoost, LightGBM, or SVM for potentially better performance.
4. **Explainability**: Integrate SHAP (SHapley Additive exPlanations) to provide feature-level explanations for individual predictions.
5. **Cross-validation**: Replace single train/test split with k-fold cross-validation for more statistically reliable evaluation.
6. **Feature engineering**: Create new features such as charge-per-month ratio or tenure buckets to improve model accuracy.
7. **Cloud deployment**: Package the Streamlit app for deployment on Streamlit Cloud or Heroku for wider accessibility.

---

## References

1. IBM Telco Customer Churn Dataset — Kaggle: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
2. Scikit-learn Documentation: https://scikit-learn.org/stable/
3. Streamlit Documentation: https://docs.streamlit.io/
4. Géron, A. (2022). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (3rd ed.). O'Reilly Media.
5. Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.

---

*This report was prepared as part of the AICTE-IBM Machine Learning Internship programme.*

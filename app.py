"""
app.py
------
Streamlit web application for Customer Churn Prediction.

Title: Customer Churn Prediction
Subtitle: Predict whether a telecom customer is likely to churn.

Usage (from customer-churn-prediction/ directory):
    streamlit run app.py
"""

import os
import sys
import joblib
import pandas as pd
import streamlit as st

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load the saved pipeline ───────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "best_model_pipeline.pkl")


@st.cache_resource
def load_pipeline():
    """Load the saved model pipeline once and cache it."""
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


pipeline = load_pipeline()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("📡 Customer Churn Prediction")
st.markdown("**Predict whether a telecom customer is likely to churn.**")
st.markdown("---")

if pipeline is None:
    st.error(
        "⚠️ **Model not found.**  \n"
        "Please train the model first by running:  \n"
        "```\npython src/train.py\n```"
    )
    st.stop()

# ── Get model info ────────────────────────────────────────────────────────────
model_name = type(pipeline.named_steps["classifier"]).__name__

# ── Sidebar — About ───────────────────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown(
        f"""
        **Project:** AICTE-IBM ML Internship  
        **Dataset:** IBM Telco Customer Churn  
        **Model:** {model_name}  

        ---
        **How to use:**
        1. Fill in customer details on the left.
        2. Click **Predict Churn**.
        3. See the prediction and probability.
        """
    )
    st.header("📘 Feature Guide")
    st.markdown(
        """
        - **Tenure:** Months the customer has been with the company.  
        - **Monthly Charges:** Amount billed per month.  
        - **Total Charges:** Total amount billed to date.  
        - **Contract:** Type of subscription agreement.  
        - **Internet Service:** Type of internet connection.  
        - **Payment Method:** How the customer pays their bill.  
        """
    )

# ── Input form ────────────────────────────────────────────────────────────────
st.subheader("👤 Enter Customer Details")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Account Info**")
    tenure = st.slider("Tenure (months)", min_value=0, max_value=72, value=12,
                        help="How long has the customer been with the company?")
    monthly_charges = st.slider("Monthly Charges ($)", min_value=18.0, max_value=120.0,
                                  value=65.0, step=0.5)
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=9000.0,
                                     value=780.0, step=10.0)
    senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Has Partner", ["No", "Yes"])
    dependents = st.selectbox("Has Dependents", ["No", "Yes"])
    gender = st.selectbox("Gender", ["Male", "Female"])

with col2:
    st.markdown("**Service Info**")
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["No", "Yes"])
    online_backup = st.selectbox("Online Backup", ["No", "Yes"])
    device_protection = st.selectbox("Device Protection", ["No", "Yes"])
    tech_support = st.selectbox("Tech Support", ["No", "Yes"])

with col3:
    st.markdown("**Billing Info**")
    streaming_tv = st.selectbox("Streaming TV", ["No", "Yes"])
    streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)",
         "Credit card (automatic)"]
    )

st.markdown("---")

# ── Predict button ────────────────────────────────────────────────────────────
predict_clicked = st.button("🔮 Predict Churn", type="primary", use_container_width=False)

if predict_clicked:
    # Build input DataFrame matching training feature order
    # Binary Yes/No fields are left as strings — the pipeline's OHE handles them
    input_data = {
        "tenure":            [tenure],
        "MonthlyCharges":    [monthly_charges],
        "TotalCharges":      [total_charges],
        "InternetService":   [internet_service],
        "Contract":          [contract],
        "PaymentMethod":     [payment_method],
        "MultipleLines":     [multiple_lines],
        "gender":            [gender],
        "Partner":           [partner],
        "Dependents":        [dependents],
        "PhoneService":      [phone_service],
        "OnlineSecurity":    [online_security],
        "OnlineBackup":      [online_backup],
        "DeviceProtection":  [device_protection],
        "TechSupport":       [tech_support],
        "StreamingTV":       [streaming_tv],
        "StreamingMovies":   [streaming_movies],
        "PaperlessBilling":  [paperless_billing],
        "SeniorCitizen":     [1 if senior_citizen == "Yes" else 0],
    }

    input_df = pd.DataFrame(input_data)

    # Generate prediction and probability
    prediction = pipeline.predict(input_df)[0]
    probability = pipeline.predict_proba(input_df)[0][1]  # probability of churn (class 1)

    # ── Display result ────────────────────────────────────────────────────────
    st.subheader("📊 Prediction Result")
    res_col1, res_col2 = st.columns(2)

    with res_col1:
        if prediction == 1:
            st.error(
                f"### ⚠️ Customer Likely to Churn\n"
                f"**Churn Probability: {probability:.1%}**"
            )
        else:
            st.success(
                f"### ✅ Customer Likely to Stay\n"
                f"**Churn Probability: {probability:.1%}**"
            )

    with res_col2:
        # Visual probability bar
        st.markdown("**Churn Probability Gauge**")
        st.progress(float(probability))
        st.caption(f"The model is {probability:.1%} confident this customer will churn.")

    # ── Simple explanation ────────────────────────────────────────────────────
    st.subheader("💡 What Influences This Prediction?")

    explanations = []
    if contract == "Month-to-month":
        explanations.append("📋 Month-to-month contracts are the strongest predictor of churn.")
    if tenure <= 12:
        explanations.append(f"📅 Short tenure ({tenure} months) — new customers are more likely to leave.")
    if monthly_charges > 70:
        explanations.append(f"💰 High monthly charges (${monthly_charges:.0f}) increase churn risk.")
    if internet_service == "Fiber optic":
        explanations.append("🌐 Fiber optic customers churn more frequently than DSL customers.")
    if payment_method == "Electronic check":
        explanations.append("💳 Electronic check payers have the highest churn rate.")
    if online_security == "No":
        explanations.append("🔒 No Online Security — customers without extra services churn more.")
    if tech_support == "No":
        explanations.append("🛠️ No Tech Support subscription detected.")
    if contract in ["One year", "Two year"]:
        explanations.append(f"✅ {contract} contract significantly reduces churn likelihood.")
    if tenure > 36:
        explanations.append(f"✅ Long tenure ({tenure} months) — loyal customers tend to stay.")

    if explanations:
        for item in explanations:
            st.markdown(f"- {item}")
    else:
        st.markdown("No specific risk factors identified for this customer profile.")

    st.caption(f"Model used: **{model_name}** | Prediction based on 19 customer features.")

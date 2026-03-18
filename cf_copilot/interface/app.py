# TEST
# 1) streamlit run cf_copilot/interface/app.py
# 2)when cloud run integrated : BACKEND_URL="https://your-cloud-run-url" streamlit run cf_copilot/interface/app.py

import os
import json
from typing import Dict, Any

import requests
import pandas as pd
import streamlit as st

# Base URL of the FastAPI backend (no /predict suffix)
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
API_URL = f"{BACKEND_URL}/predict"

def call_api(payload: Dict[str, Any]) -> Dict[str, Any]:
    response = requests.post(API_URL, json=payload)
    response.raise_for_status()
    return response.json()

st.set_page_config(page_title="Cashflow Copilot", layout="centered")

st.title("Cashflow Copilot – Invoice Payment Risk")

st.markdown(
    "Fill in the invoice features below and click **Predict** "
    "to see the expected payment bucket and probabilities."
)

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        invoice_age_days = st.number_input("Invoice age (days)", min_value=0, value=30)
        days_until_due = st.number_input("Days until due", min_value=-365, value=10)
        days_past_due = st.number_input("Days past due", min_value=-365, value=0)
        invoice_month = st.number_input("Invoice month", 1, 12, 3)
        due_month = st.number_input("Due month", 1, 12, 4)
        days_since_last_invoice = st.number_input(
            "Days since last invoice", min_value=-1, value=20
        )
        invoice_amount = st.number_input(
            "Invoice amount", min_value=0.0, value=5000.0, step=100.0
        )

    with col2:
        customer_avg_delay = st.number_input(
            "Customer avg delay (days)", min_value=0.0, value=2.5
        )
        late_payment_ratio = st.number_input(
            "Late payment ratio", min_value=0.0, max_value=1.0, value=0.3
        )
        prev_transaction_count = st.number_input(
            "Previous transaction count", min_value=0, value=15
        )
        customer_risk_score = st.number_input(
            "Customer risk score", min_value=0.0, max_value=1.0, value=0.6
        )
        invoice_currency = st.text_input("Invoice currency", value="USD")
        document_type = st.text_input("Document type", value="RV")
        cust_payment_terms = st.text_input("Payment terms", value="N30")

    submitted = st.form_submit_button("Predict")

if submitted:
    payload = {
        "invoice_age_days": invoice_age_days,
        "days_until_due": days_until_due,
        "days_past_due": days_past_due,
        "invoice_month": invoice_month,
        "due_month": due_month,
        "customer_avg_delay": customer_avg_delay,
        "late_payment_ratio": late_payment_ratio,
        "prev_transaction_count": prev_transaction_count,
        "customer_risk_score": customer_risk_score,
        "days_since_last_invoice": days_since_last_invoice,
        "invoice_amount": invoice_amount,
        "invoice_currency": invoice_currency,
        "document_type": document_type,
        "cust_payment_terms": cust_payment_terms,
    }

    try:
        result = call_api(payload)
    except requests.HTTPError as e:
        st.error(f"API error: {e}")
    except Exception as e:
        st.error(f"Unexpected error: {e}")
    else:
        if "error" in result:
            st.error(result["error"])
        else:
            st.subheader(f"Predicted bucket: {result['predicted_bucket']}")

            probs = result.get("probabilities", {})
            if probs:
                df_probs = pd.DataFrame(
                    {"bucket": list(probs.keys()), "probability": list(probs.values())}
                ).sort_values("bucket")

                st.bar_chart(
                    df_probs.set_index("bucket")["probability"],
                    height=300,
                )

            with st.expander("Raw API response"):
                st.json(result)

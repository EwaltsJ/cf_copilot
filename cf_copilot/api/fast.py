# TEST :
"""
   curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "invoice_age_days": 30,
    "days_until_due": 10,
    "days_past_due": 0,
    "invoice_month": 3,
    "due_month": 4,
    "customer_avg_delay": 2.5,
    "late_payment_ratio": 0.3,
    "prev_transaction_count": 15,
    "customer_risk_score": 0.6,
    "days_since_last_invoice": 20,
    "invoice_amount": 5000.0,
    "invoice_currency": "USD",
    "document_type": "RV",
    "cust_payment_terms": "N30"
  }'
"""
"""
{"predicted_bucket":1,"probabilities":{"1":0.24848501830215458,"2":\
    0.20760917237606522,"3":0.139970470155515,"4":0.11466307744251168,"5":0.0849923725278862,"6":0.09047090008642361,\
        "7":0.11380898910944395}}%
"""
from typing import Dict, Any

import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from cf_copilot.ml_logic.registry import load_model
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class InvoiceFeatures(BaseModel):
    invoice_age_days: float
    days_until_due: float
    days_past_due: float
    invoice_month: int
    due_month: int
    customer_avg_delay: float
    late_payment_ratio: float
    prev_transaction_count: int
    customer_risk_score: float
    days_since_last_invoice: float
    invoice_amount: float
    invoice_currency: str
    document_type: str
    cust_payment_terms: str


app = FastAPI(title="Cashflow Copilot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once at startup
model = load_model()

@app.get("/")
def root():
    return {"message": "Cashflow Copilot API is running"}


@app.post("/predict")
def predict_invoice(features: InvoiceFeatures) -> Dict[str, Any]:
    global model
    if model is None:
        return {"error": "No model available. Train and save a model first."}

    df = pd.DataFrame([features.model_dump()])

    probas = model.predict_proba(df)[0]
    pred = model.predict(df)[0]

    logger.info("Prediction made for customer features: %s -> bucket %s", features.model_dump(), pred)

    prob_dict = {str(cls): float(p) for cls, p in zip(model.classes_, probas)}

    return {
        "predicted_bucket": int(pred),
        "probabilities": prob_dict,
    }

@app.get("/health")
def health():
    """
    Lightweight health check.
    """
    global model
    return {
        "status": "ok",
        "model_loaded": model is not None,
    }


@app.get("/version")
def version():
    """
    Simple static version for debugging / clients.
    """
    return {"version": "0.1.0"}

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from io import BytesIO

from cf_copilot.ml_logic.registry import load_model, predict
from cf_copilot.ml_logic.data import load_cashflow_data, data_cleaning, engineer_features
from cf_copilot.ml_logic.encoders import preprocess

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model after the server has started (not at import time)
app.state.pipeline = load_model()

@app.get("/")
def root():
    return {"message": "Hi, The API is running!"}


@app.post("/predict")
async def post_predict(file: UploadFile = File(...)):
    """Accept a CSV of invoices and return week-bucket predictions.

    The CSV should contain the same columns as the raw invoice data
    (cust_number, due_in_date, invoice_currency, document_type,
    total_open_amount, baseline_create_date, cust_payment_terms, etc.).
    """
    pipeline = app.state.pipeline

    if pipeline is None:
        return {"error": "No trained model found. Run train() first."}

    contents = await file.read()
    df = pd.read_csv(BytesIO(contents))

    return predict(pipeline, df)

    return {"predictions": predictions}


@app.get("/debug-load-data")
def debug_load_data():
    df = load_cashflow_data()
    return {
        "rows": len(df),
        "cols": list(df.columns),
    }

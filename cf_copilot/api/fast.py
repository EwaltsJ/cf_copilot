from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from io import BytesIO

from cf_copilot.ml_logic.registry import load_model, predict
from cf_copilot.ml_logic.data import load_cashflow_data
from cf_copilot.cashflow_prediction.registry import predict_cashflow

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

    results = predict(pipeline, df)

    buckets = [int(b) for b in pipeline.classes_]
    bucket_labels = [f"week_{b}" for b in buckets]

    predictions = [
        {
            "invoice_id": int(df.iloc[i]["invoice_id"]),
            "predicted_bucket": int(results["week_bucket"][i]),
            "bucket_probabilities": dict(zip(bucket_labels, map(lambda p: round(float(p), 4), results["probabilities"][i]))),
        }
        for i in range(len(results["week_bucket"]))
    ]

    return {"predictions": predictions}

@app.post("/predict_cashflow")
async def post_predict_cashflow(file: UploadFile = File(...)):
    """Accept a CSV of invoices and return cashflow-forecast for the upcoming 6 weeks.

    The CSV should contain the same columns as the raw invoice data
    (cust_number, due_in_date, invoice_currency, document_type,
    total_open_amount, baseline_create_date, cust_payment_terms, etc.).
    """
    pipeline = app.state.pipeline

    contents = await file.read()
    df = pd.read_csv(BytesIO(contents))

    weekly_forecast_df = predict_cashflow(df, pipeline)

    return weekly_forecast_df.to_dict(orient="records")

@app.get("/debug-load-data")
def debug_load_data():
    df = load_cashflow_data()
    return {
        "rows": len(df),
        "cols": list(df.columns),
    }

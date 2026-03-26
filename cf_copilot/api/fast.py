from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from io import BytesIO
from contextlib import asynccontextmanager
from pathlib import Path

from cf_copilot.ml_logic.registry import load_model, predict
from cf_copilot.ml_logic.data import load_cashflow_data, load_historical_data
from cf_copilot.cashflow_prediction.registry import predict_cashflow
from cf_copilot.collection_ranking.invoices_ranker import get_priority_invoices

from cf_copilot.rag.script_generator import generate_script, load_vector_store
from cf_copilot.params import CHROMA_PATH, CURRENT_DATE

@asynccontextmanager
async def lifespan(app):
    """Load model and vector store once when the server starts"""
    # Load ML pipeline
    try:
        app.state.pipeline = load_model()
        print("✅ ML model loaded successfully.")
    except Exception as e:
        print(f"⚠️  Model load failed at startup: {e}")
        app.state.pipeline = None

    # Load RAG vector store
    try:
        if not CHROMA_PATH.exists():
            print(f"⚠️ Chroma path not found: {CHROMA_PATH}")
            app.state.vector_store = None
        else:
            app.state.vector_store = load_vector_store(CHROMA_PATH)
            print("✅ Vector store loaded successfully.")
    except Exception as e:
        print(f"⚠️ Vector store load failed at startup: {e}")
        app.state.vector_store = None

    yield
    # Shutdown (nothing to clean up)

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Hi, The API is running!"}


@app.post("/predict")
async def post_predict(file: UploadFile = File(...)):
    """Return per-invoice week-bucket predictions with probabilities."""
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
            "doc_id": int(df.iloc[i]["doc_id"]),
            "predicted_bucket": int(results["week_bucket"][i]),
            "bucket_probabilities": dict(zip(
                bucket_labels,
                [round(float(p), 4) for p in results["probabilities"][i]],
            )),
        }
        for i in range(len(results["week_bucket"]))
    ]

    return {"predictions": predictions}


@app.post("/predict_cashflow")
async def post_predict_cashflow(file: UploadFile = File(...)):
    """Return aggregated weekly cashflow forecast."""
    pipeline = app.state.pipeline

    if pipeline is None:
        return {"error": "No trained model found. Run train() first."}

    contents = await file.read()
    df = pd.read_csv(BytesIO(contents))

    weekly_forecast_df = predict_cashflow(df, pipeline)

    return weekly_forecast_df.to_dict(orient="records")


@app.post("/prioritise_invoices")
async def post_get_priority_invoices(
    file: UploadFile = File(...),
):
    """Return top 10 risky invoices to prioritise for collection."""
    pipeline = app.state.pipeline

    if pipeline is None:
        return {"error": "No trained model found. Run train() first."}

    contents = await file.read()
    df = pd.read_csv(BytesIO(contents))

    try:
        current_date_parsed = CURRENT_DATE
    except Exception:
        return {"error": "Invalid current date format. Use YYYY-MM-DD"}

    priority_invoices_df = get_priority_invoices(df, pipeline, current_date_parsed)

    return priority_invoices_df.to_dict(orient="records")


@app.post("/rag_script")
async def post_rag_script(invoice: dict):
    """
    Generate a RAG/LLM collection email + action script for one invoice.
    Expects a JSON body with invoice fields.
    """
    vector_store = app.state.vector_store

    if vector_store is None:
        return {
            "error": "Vector store not loaded. Check CHROMA path and startup logs."
        }

    invoice_data = load_historical_data()
    current_invoice = invoice_data[invoice_data['doc_id'] == int(invoice['doc_id'])]

    if current_invoice.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Invoice {invoice['doc_id']} not found in historical data."
        )

    # Compute days_past_due from due_in_date
    current_invoice["days_past_due"] = (pd.Timestamp.now() - pd.to_datetime(current_invoice["due_in_date"])).dt.days
    # Enrich the incoming dict with required fields from the DataFrame
    row = current_invoice.iloc[0]
    for field in ["doc_id", "name_customer", "cust_number", "total_open_amount", "due_in_date", "days_past_due"]:
        if field not in invoice and field in row.index:
            invoice[field] = row[field]

    required_fields = [
        "doc_id",
        "name_customer",
        "cust_number",
        "total_open_amount",
        "due_in_date",
        "days_past_due",
    ]
    missing_fields = [field for field in required_fields if field not in invoice]
    if missing_fields:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required invoice fields: {missing_fields}"
        )

    try:
        result = generate_script(invoice=invoice, vector_store=vector_store)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate RAG script: {str(e)}"
        )

@app.get("/debug-load-data")
def debug_load_data():
    df = load_cashflow_data()
    return {"rows": len(df), "cols": list(df.columns)}

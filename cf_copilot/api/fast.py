from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from io import BytesIO
from contextlib import asynccontextmanager

from cf_copilot.ml_logic.registry import load_model, predict
from cf_copilot.ml_logic.data import load_cashflow_data
from cf_copilot.cashflow_prediction.registry import predict_cashflow
from cf_copilot.collection_ranking.invoices_ranker import get_priority_invoices


@asynccontextmanager
async def lifespan(app):
    """Load model once the server is running"""
    try:
        app.state.pipeline = load_model()
    except Exception as e:
        print(f"⚠️  Model load failed at startup: {e}")
        app.state.pipeline = None
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
    current_date: str = Form(...)
):
    """Return top 10 risky invoices to prioritise for collection."""
    pipeline = app.state.pipeline

    if pipeline is None:
        return {"error": "No trained model found. Run train() first."}

    contents = await file.read()
    df = pd.read_csv(BytesIO(contents))

    try:
        current_date_parsed = pd.to_datetime(current_date)
    except Exception:
        return {"error": "Invalid current date format. Use YYYY-MM-DD"}

    priority_invoices_df = get_priority_invoices(df, pipeline, current_date_parsed)

    return priority_invoices_df.to_dict(orient="records")


@app.get("/debug-load-data")
def debug_load_data():
    df = load_cashflow_data()
    return {"rows": len(df), "cols": list(df.columns)}

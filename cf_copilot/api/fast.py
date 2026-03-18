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

# Load the trained pipeline once at startup
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

    # Clean and engineer features for the uploaded invoices
    model_df, _ = data_cleaning(df)
    current_date = pd.Timestamp.now()
    featured_df = engineer_features(model_df, model_df, current_date)

    X, _ = preprocess(featured_df)

    results = predict(pipeline, X)

    predictions = []
    for i in range(len(X)):
        predictions.append({
            "week_bucket": int(results["week_bucket"][i]),
            "probabilities": results["probabilities"][i].tolist(),
        })

    return {"predictions": predictions}


@app.get("/debug-load-data")
def debug_load_data():
    df = load_cashflow_data()
    return {
        "rows": len(df),
        "cols": list(df.columns),
    }

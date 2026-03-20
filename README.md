# CF Copilot — Invoice Payment Prediction Platform

CF Copilot is an end-to-end platform that lets finance teams **upload invoices and instantly predict when they'll be paid**, bucketed into weekly intervals (1–7 weeks). Upload a batch of invoices, get back payment-timing predictions with probability scores, and use the insights to forecast cash flow with confidence.

## What It Does

1. **Upload** — Send invoice data to the prediction API.
2. **Predict** — The platform engineers features on the fly, runs them through a trained Random Forest pipeline, and returns a predicted payment week (1–7) for each invoice along with class probabilities.
3. **Plan** — Use the predictions to prioritise collections, forecast liquidity, and flag late-payment risk before it materialises.

```
Upload invoices  →  Feature engineering  →  ML prediction  →  Week bucket + probabilities
```

## How It Works

### Data Pipeline

Raw invoice data is cleaned, deduplicated, and enriched with engineered features built from a sliding-window approach over historical payment records. Each weekly snapshot captures every open invoice at that point in time, creating a rich augmented training set.

### Feature Engineering

Features are computed for every invoice and fall into three groups:

- **Invoice timing** — age in days, days until/past due, pay-term length, calendar months.
- **Customer behaviour** — average payment delay, late-payment ratio, transaction count, days since last invoice, composite risk score.
- **Amount characteristics** — open amount, log-transformed amount, size category (small / medium / large).

### Model

An `sklearn` Pipeline chains a `ColumnTransformer` (median imputation for numerics, ordinal encoding for categoricals) with a tuned `RandomForestClassifier`. Training uses a time-based 80/20 split and walk-forward backtesting validates stability across temporal cutoffs.

## Project Structure

```
├── cf_copilot/
│   ├── __init__.py
│   ├── params.py                # Project constants (LOCAL_REGISTRY_PATH, etc.)
│   ├── utils.py                 # Shared utilities
│   ├── api/
│   │   ├── __init__.py
│   │   └── fast.py              # FastAPI endpoints (health check, predict)
│   ├── dashboard/
│   │   └── cf_copilot_dashboard_v02.py  # Streamlit dashboard
│   ├── interface/
│   │   └── main.py              # CLI entrypoint — train() and pred() functions
│   └── ml_logic/
│       ├── data.py              # Data loading, cleaning, feature engineering, sliding windows
│       ├── encoders.py          # Feature/target split and preprocessing
│       ├── model.py             # Pipeline init, training, evaluation, calibration, backtesting
│       └── registry.py          # Save / load models (pickle), run predictions
├── notebooks/                   # Exploration & preprocessing notebooks
├── raw_data/                    # Downloaded and processed CSVs (created at runtime)
├── tests/                       # API endpoint tests and project structure checks
├── scripts/
├── Dockerfile                   # API container
├── Dockerfile.streamlit         # Dashboard container
├── docker-compose.yml           # Local multi-service orchestration
├── Makefile                     # Common commands (train, run, test, etc.)
├── requirements.txt
├── requirements_dev.txt
└── setup.py
```

## Getting Started

### Prerequisites

- Python 3.10+
- Docker & Docker Compose
- Kaggle credentials (for initial dataset download — optional if you supply your own data)
- A GCP service account key with access to your GCS bucket

### Installation

```bash
git clone <repo-url> && cd cf_copilot
pip install -e .
```

### Environment

```bash
cp .env.example .env
# Fill in GCP_PROJECT_ID, GCS_BUCKET_NAME, MLFLOW_* and other values
```

Make sure `GOOGLE_APPLICATION_CREDENTIALS` is set in your shell to your GCP service account key:

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/your/service-account-key.json
```

### Training

Run the full pipeline (load → clean → augment → train → evaluate → save):

```bash
python -m cf_copilot.interface.main
```

Or from Python:

```python
from cf_copilot.interface.main import train
pipeline = train()
```

### Running the API locally

```bash
uvicorn cf_copilot.api.fast:app --reload
```

**Endpoints:**

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `GET` | `/predict` | Return payment-week predictions for input features |
| `GET` | `/debug-load-data` | Sanity check — returns row count and column names |

### Running the Dashboard locally

```bash
streamlit run cf_copilot/dashboard/cf_copilot_dashboard_v02.py
```

---

## Docker

### API only

```bash
docker build -t cf-copilot .
docker run -p 8080:8080 cf-copilot
```

### API + Dashboard together (recommended)

```bash
# First build — or after any code change:
docker compose up --build

# Subsequent runs (no code changes):
docker compose up

# Force a clean rebuild with no cached layers:
docker compose build --no-cache && docker compose up

# Tear down containers:
docker compose down
```

| Service | URL |
|---|---|
| API | http://localhost:8080 |
| Dashboard | http://localhost:8501 |

> **Note:** `GOOGLE_APPLICATION_CREDENTIALS` must be set in your shell before running `docker compose` — the key file is mounted read-only into each container at runtime and is never baked into the image.

---

## Making Predictions

Once the model is trained and the API is running:

```bash
curl "http://localhost:8080/predict?input_one=154&input_two=199"
```

Each invoice receives a predicted **week bucket** (1 = paid within the first week, 7 = paid in week 7 or later) and a probability distribution across all seven buckets.

## Key Dependencies

| Package | Purpose |
|---|---|
| scikit-learn | Pipeline, Random Forest, metrics |
| pandas / NumPy | Data wrangling & feature engineering |
| FastAPI / Uvicorn | Prediction API |
| Streamlit | Interactive dashboard |
| kagglehub | Dataset download |
| MLflow | Experiment tracking |
| LangChain / ChromaDB | RAG pipeline for AI email generation |
| matplotlib | Calibration curves & visualisation |

## Training Data

The model is trained on the [Payment Date Prediction for Invoices](https://www.kaggle.com/datasets/pradumn203/payment-date-prediction-for-invoices-dataset) dataset from Kaggle. The pipeline downloads it automatically on first run if `raw_data/dataset.csv` is not already present.

## Testing

```bash
pytest tests/
```

## License

See `LICENSE` for details.

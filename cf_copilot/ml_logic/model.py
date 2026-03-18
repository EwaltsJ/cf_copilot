import time
import pickle
import numpy as np

from colorama import Fore, Style
from typing import Tuple

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, log_loss

NUMERIC_FEATURES = [
    "invoice_age_days", "days_until_due", "days_past_due",
    "invoice_month", "due_month",
    "customer_avg_delay", "late_payment_ratio",
    "prev_transaction_count", "customer_risk_score",
    "days_since_last_invoice", "open_amount",
]

CATEGORICAL_FEATURES = [
    "invoice_currency", "document_type", "cust_payment_terms",
]


def initialize_model() -> Pipeline:
    """
    Initialize a RandomForestClassifier inside a sklearn Pipeline
    with a ColumnTransformer preprocessor.
    """

    # Numeric: fill NaN with median
    numeric_transformer = SimpleImputer(strategy="median")

    # Categorical: fill NaN then ordinal encode
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value=-1)),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    classifier = RandomForestClassifier(
        n_estimators=347,
        max_depth=21,
        min_samples_split=8,
        min_samples_leaf=4,
        max_features=0.267,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])

    print("✅ Model (pipeline) initialized")
    return pipeline


def train_model(
        model: Pipeline,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Pipeline:
    """
    Fit the pipeline and return the fitted model.
    """
    print(Fore.BLUE + "\nTraining model..." + Style.RESET_ALL)

    model.fit(X, y)

    print(f"✅ Model trained on {len(X)} rows")
    return model


def evaluate_model(
        model: Pipeline,
        X: np.ndarray,
        y: np.ndarray,
    ) -> dict:
    """
    Evaluate trained model performance on the dataset.
    Returns a dict with log_loss and classification_report.
    """
    print(Fore.BLUE + f"\nEvaluating model on {len(X)} rows..." + Style.RESET_ALL)

    if model is None:
        print(f"\n❌ No model to evaluate")
        return None

    probas = model.predict_proba(X)
    preds = model.predict(X)

    ll = log_loss(y, probas, labels=model.classes_)
    report = classification_report(y, preds)

    print(f"✅ Model evaluated, log_loss: {round(ll, 4)}")
    print(report)

    return {"log_loss": ll}

import numpy as np

from colorama import Fore, Style

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBClassifier
from sklearn.feature_selection import VarianceThreshold

NUMERIC_FEATURES = [
    "business_year", "invoice_age_days", "days_until_due", "pay_terms_days",
    "invoice_month", "due_month", "days_past_due", "customer_avg_delay",
    "late_payment_ratio", "prev_transaction_count", "days_since_last_invoice",
    "customer_risk_score", "invoice_amount", "invoice_amount_log",
]

CATEGORICAL_FEATURES = [
    "invoice_currency", "document_type", "invoice_size_cat", "invoice_size_cat_q"
]

def initialize_model() -> Pipeline:
    """Initialize a RandomForestClassifier inside a sklearn Pipeline.

    The pipeline includes a ColumnTransformer preprocessor that imputes
    numeric features with median and ordinal-encodes categorical features.

    Returns:
        An unfitted sklearn Pipeline.
    """
    numeric_transformer = SimpleImputer(strategy="median")

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

    classifier = XGBClassifier(
        colsample_bytree=0.727,
        gamma=0.059,
        learning_rate=0.025,
        max_depth=7,
        min_child_weight=2,
        n_estimators=352,
        reg_alpha=0.587,
        reg_lambda=1.931,
        subsample=0.882,
        tree_method="hist",
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("variance", VarianceThreshold(threshold=0.05)),
        ("classifier", classifier),
    ])

    print("✅ Model (pipeline) initialized")
    return pipeline


def train_model(model: Pipeline, X: np.ndarray, y: np.ndarray) -> Pipeline:
    """Fit the pipeline and return the fitted model.

    Args:
        model: an unfitted sklearn Pipeline.
        X: feature DataFrame.
        y: target Series.

    Returns:
        The fitted Pipeline.
    """
    print(Fore.BLUE + "\nTraining model..." + Style.RESET_ALL)

    model.fit(X, y)

    print(f"✅ Model trained on {len(X)} rows")
    return model

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
    "invoice_month_sin","invoice_month_cos","due_month_sin","due_month_cos"
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
        ("imputer", SimpleImputer(strategy="most_frequent", fill_value=-1)),
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
        objective="multi:softprob",
        num_class=7,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=4,
        tree_method="hist",
        verbosity=0,
        colsample_bytree=0.62625093750534,
        gamma=0.12896652080446278,
        learning_rate=0.025805236304460133,
        max_depth=9,
        min_child_weight=6,
        n_estimators=264,
        reg_alpha=1.1682580031634093,
        reg_lambda=1.7185239911639494,
        subsample=0.7919464969390382,
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
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

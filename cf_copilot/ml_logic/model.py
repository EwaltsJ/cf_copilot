import numpy as np

from colorama import Fore, Style

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier

NUMERIC_FEATURES = [
    "invoice_age_days","days_until_due","pay_terms_days","customer_avg_delay",
    "days_since_last_invoice","invoice_amount_log","invoice_month_sin",
]

def initialize_model() -> Pipeline:
    """Initialize a RandomForestClassifier inside a sklearn Pipeline.

    The pipeline includes a ColumnTransformer preprocessor that imputes
    numeric features with median.

    Returns:
        An unfitted sklearn Pipeline.
    """
    numeric_transformer = SimpleImputer(strategy="median")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
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

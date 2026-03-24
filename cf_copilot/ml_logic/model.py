import numpy as np

from colorama import Fore, Style

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier

from cf_copilot.ml_logic.encoders import NUMERIC_FEATURES, CATEGORICAL_FEATURES


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

    classifier = RandomForestClassifier(
        n_estimators=467,
        max_depth=18,
        min_samples_split=6,
        min_samples_leaf=2,
        max_features= 0.35,
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

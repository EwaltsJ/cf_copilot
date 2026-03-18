import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from colorama import Fore, Style

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, log_loss, confusion_matrix
from sklearn.calibration import calibration_curve

from cf_copilot.ml_logic.encoders import NUMERIC_FEATURES, CATEGORICAL_FEATURES, preprocess


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
        n_estimators=400,
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=4,
        max_features=0.3,
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


def evaluate_model(model: Pipeline, X: np.ndarray, y: np.ndarray) -> dict:
    """Evaluate trained model performance on the dataset.

    Prints log loss, classification report, and confusion matrix.

    Args:
        model: a fitted sklearn Pipeline.
        X: feature DataFrame.
        y: true target labels.

    Returns:
        A dict with 'log_loss' score.
    """
    print(Fore.BLUE + f"\nEvaluating model on {len(X)} rows..." + Style.RESET_ALL)

    if model is None:
        print("❌ No model to evaluate")
        return None

    probas = model.predict_proba(X)
    preds = model.predict(X)

    ll = log_loss(y, probas, labels=model.classes_)
    print(f"✅ Model evaluated, log_loss: {round(ll, 4)}")
    print(classification_report(y, preds))
    print(confusion_matrix(y, preds))

    return {"log_loss": ll}


def show_calibration_curves(probas, pipeline, y_test):
    """Plot one-vs-rest calibration curves for each target bucket.

    Args:
        probas: predicted class probabilities of shape (n_samples, n_classes).
        pipeline: fitted estimator with a classes_ attribute.
        y_test: true class labels.
    """
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for i, bucket in enumerate(pipeline.classes_):
        ax = axes.flat[i]
        y_binary = (y_test == bucket).astype(int)
        prob_true, prob_pred = calibration_curve(y_binary, probas[:, i], n_bins=10)
        ax.plot(prob_pred, prob_true, marker="o")
        ax.plot([0, 1], [0, 1], "k--")
        ax.set_title(f"Bucket {bucket}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.show()


def simulate_past_performance(pipeline, df):
    """Run a walk-forward backtest of the pipeline over historical cutoffs.

    Args:
        pipeline: an unfitted sklearn Pipeline.
        df: full DataFrame with 'reference_date' column for temporal splitting.
    """
    reference_dates = df["reference_date"].sort_values().unique()
    cutoffs = np.percentile(reference_dates.astype(int), [40, 50, 60, 70, 80])
    cutoffs = pd.to_datetime(cutoffs)

    scores = []
    for cutoff in cutoffs:
        train_df = df[df["reference_date"] <= cutoff]
        test_df = df[
            (df["reference_date"] > cutoff) &
            (df["reference_date"] <= cutoff + pd.Timedelta(weeks=6))
        ]

        if len(test_df) == 0:
            continue

        X_train, y_train = preprocess(train_df)
        X_test, y_test = preprocess(test_df)

        pipeline.fit(X_train, y_train)
        probas = pipeline.predict_proba(X_test)
        score = log_loss(y_test, probas, labels=pipeline.classes_)
        scores.append(score)
        print(f"Cutoff {cutoff.date()} -> log_loss: {score:.4f}")

    print(f"\nAverage: {np.mean(scores):.4f} (std: {np.std(scores):.4f})")

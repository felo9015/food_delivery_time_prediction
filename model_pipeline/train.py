"""End-to-end training script for the final delivery-time prediction pipeline.

Runs the pre-freeze cross-validation stability check on the training split,
then refits the final pipeline on 100% of the data and serializes it. See
model_notes.md, "Selected Model and Production Pipeline," for the reasoning
behind the model choice and the retrain-on-full-data decision.

Usage:
    python -m model_pipeline.train
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from model_pipeline.config import RANDOM_STATE, TEST_SIZE
from model_pipeline.data_preprocessing import build_preprocessing_pipeline

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "Food_Delivery_Times.csv"
ARTIFACT_PATH = Path(__file__).resolve().parent / "artifacts" / "model.joblib"


def build_final_pipeline() -> Pipeline:
    """Preprocessing + LinearRegression, the model selected in model_notes.md."""
    return Pipeline([
        ("preprocessing", build_preprocessing_pipeline()),
        ("model", LinearRegression()),
    ])


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["Order_ID", "Delivery_Time_min"])
    y = df["Delivery_Time_min"]

    # Stability check: 5-fold CV MAE on the same 80/20 training split used
    # throughout model_notes.md, as a second, independent read on expected
    # performance before the model is frozen.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    cv_pipeline = build_final_pipeline()
    cv_mae_scores = -cross_val_score(
        cv_pipeline, X_train, y_train, cv=5, scoring="neg_mean_absolute_error"
    )
    print(f"5-fold CV MAE: {cv_mae_scores.mean():.3f} +/- {cv_mae_scores.std():.3f}")

    cv_pipeline.fit(X_train, y_train)
    test_mae = mean_absolute_error(y_test, cv_pipeline.predict(X_test))
    print(f"Held-out test MAE (80/20 split): {test_mae:.3f}")

    # Production pipeline: same architecture, refit on 100% of the data.
    final_pipeline = build_final_pipeline()
    final_pipeline.fit(X, y)

    ARTIFACT_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(final_pipeline, ARTIFACT_PATH)
    print(f"Final pipeline retrained on 100% of the data and saved to {ARTIFACT_PATH}")


if __name__ == "__main__":
    main()

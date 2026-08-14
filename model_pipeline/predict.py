"""Load the serialized final pipeline and predict delivery time for new orders.

Usage:
    python -m model_pipeline.predict
"""

from pathlib import Path

import joblib
import pandas as pd

MODEL_PATH = Path(__file__).resolve().parent / "artifacts" / "model.joblib"


def load_model():
    """Load the serialized production pipeline (preprocessing + LinearRegression)."""
    return joblib.load(MODEL_PATH)


def predict_delivery_time(order: dict, model=None) -> float:
    """Predict delivery time, in minutes, for a single raw order.

    `order` uses the same raw column names as the source dataset —
    Distance_km, Weather, Traffic_Level, Time_of_Day, Vehicle_Type,
    Preparation_Time_min, Courier_Experience_yrs — before any imputation or
    encoding. Missing values (None) are handled by the same imputation
    fitted at training time.
    """
    if model is None:
        model = load_model()
    order_df = pd.DataFrame([order])
    return float(model.predict(order_df)[0])


if __name__ == "__main__":
    example_order = {
        "Distance_km": 7.5,
        "Weather": "Rainy",
        "Traffic_Level": "Medium",
        "Time_of_Day": "Evening",
        "Vehicle_Type": "Scooter",
        "Preparation_Time_min": 15,
        "Courier_Experience_yrs": 3.0,
    }
    prediction = predict_delivery_time(example_order)
    print(f"Predicted delivery time: {prediction:.2f} min")

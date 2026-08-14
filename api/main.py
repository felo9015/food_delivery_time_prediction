"""FastAPI service exposing the delivery-time prediction model.

Conceptual walkthrough of how this works: api/api.md
Practical setup and usage instructions: api/README.md
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from api.schemas import DeliveryOrder, PredictionResponse
from model_pipeline.predict import load_model, predict_delivery_time

# A plain dict used as a small mutable container for the loaded model object.
# `lifespan`, below, is an async context manager FastAPI runs once when the
# application starts (the code before `yield`) and once when it shuts down
# (the code after `yield`). The model is loaded there -- not inside the
# /predict endpoint -- because loading model.joblib means reading it from
# disk and deserializing a full scikit-learn pipeline object, which is real
# I/O and CPU work. Doing that on every request would repeat that cost on
# every single prediction; loading it once at startup means every request
# just reuses the same already-loaded model already sitting in memory.
model_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_state["model"] = load_model()
    yield
    model_state.clear()


app = FastAPI(
    title="Delivery Time Prediction API",
    description="Predicts food delivery time, in minutes, from order and context features.",
    lifespan=lifespan,
)


@app.post("/predict", response_model=PredictionResponse)
def predict(order: DeliveryOrder) -> PredictionResponse:
    """Predict delivery time for a single order.

    By the time this function runs, `order` has already been validated by
    Pydantic against the `DeliveryOrder` schema -- a request with a missing
    required field, a wrong type, or a category value outside the ones the
    model was trained on never reaches this function at all; FastAPI
    rejects it first with an automatic 422 response describing exactly what
    was wrong.
    """
    try:
        predicted_minutes = predict_delivery_time(order.model_dump(), model=model_state["model"])
    except Exception as exc:
        # Anything that goes wrong past validation (e.g. an unexpected
        # failure inside the model pipeline itself) is reported as an
        # explicit 500 with the actual error message, instead of leaking a
        # bare, unexplained "Internal Server Error" to the caller.
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    return PredictionResponse(predicted_delivery_time_min=predicted_minutes)

# How the API Works

This document explains how the prediction service is put together. For installation and usage instructions, see `api/README.md`.

## The Three Pieces

| Piece | Role |
|---|---|
| **FastAPI** | The web framework. It turns a plain Python function (`predict` in `api/main.py`) into an HTTP endpoint (`POST /predict`) that a client can call over the network, and generates the interactive documentation at `/docs`. |
| **Pydantic** | The validation layer. `api/schemas.py` defines what a valid request looks like (which fields, which types, which category values); FastAPI uses that definition to check every incoming request before the prediction function ever runs. |
| **uvicorn** | The server that actually runs the application — it listens for HTTP connections, reads each request, hands it to FastAPI, and sends the response back. FastAPI defines *what* happens for a request; uvicorn is the process that makes the application reachable at all. |

## Request Flow

1. A client sends a JSON request body to `POST /predict` (e.g. `{"Distance_km": 7.5, "Weather": "Rainy", ...}`).
2. **Pydantic validates it** against `DeliveryOrder` (`api/schemas.py`): every required field must be present, `Distance_km`/`Preparation_Time_min` must be positive numbers, and `Weather`/`Traffic_Level`/`Time_of_Day`/`Vehicle_Type` must be one of the exact category values the model was trained on. If anything fails, FastAPI returns a `422` response describing exactly what was wrong, and the request never reaches the prediction code.
3. **The validated data is handed to the same preprocessing pipeline used at training time** (`model_pipeline/data_preprocessing.py`, loaded as part of the serialized `model.joblib`) — the same imputation and encoding steps documented in `EDA_report.md` and `model_notes.md`, so a request is transformed exactly the way training data was.
4. **The already-loaded model produces a prediction** — a single number, the predicted delivery time in minutes.
5. **The response is returned as JSON**, validated against `PredictionResponse` (`{"predicted_delivery_time_min": 50.8}`).

## Design Decisions

- **The model is loaded once, at application startup, not inside the `/predict` function.** `model.joblib` is a serialized `scikit-learn` pipeline; loading it means reading a file from disk and reconstructing that Python object, which costs real time. Doing that once when the server starts, and keeping the loaded model in memory for the life of the process, means every request reuses the same object instead of repeating that cost on every single call. This is implemented with FastAPI's `lifespan` mechanism in `api/main.py`.
- **`Weather`, `Traffic_Level`, and `Time_of_Day` accept `"Unknown"` as a normal, valid value.** This mirrors how missing values in these columns were actually handled during training: `model_pipeline/data_preprocessing.py` imputes them with the constant category `"Unknown"` rather than a real weather/traffic/time value (`EDA_report.md`). Since the model already learned what to do with `"Unknown"`, a caller who does not have this information can send it explicitly instead of guessing a plausible-looking value that isn't true.
- **`Courier_Experience_yrs` is optional.** If the caller omits it, the same pipeline imputes it with the median learned from the training data — exactly what happens for a missing value in the training set. Making it optional at the API level avoids forcing a caller to invent a number when the field is genuinely unknown.
- **`Vehicle_Type` has no `"Unknown"` option, and is required.** `EDA_report.md` found zero missing values for this column, so the training pipeline never learned an `"Unknown"` category for it — offering that option in the API would let a request through that the model was never prepared to handle.
- **The category lists in `api/schemas.py` were read directly from the trained encoder** (`model_pipeline/artifacts/model.joblib`), not written from memory or assumption — this guarantees the API can never accept a category the model has never actually seen.
- **Prediction failures return an explicit `500` with a message, not a bare unhandled error.** Validation already rejects malformed input before it reaches the prediction code (step 2 above), but if something still fails inside the pipeline itself, the error is caught and reported with context (`api/main.py`), rather than surfacing as an opaque server error.

## Example

Request:

```json
POST /predict
{
  "Distance_km": 7.5,
  "Preparation_Time_min": 15,
  "Weather": "Rainy",
  "Traffic_Level": "Medium",
  "Time_of_Day": "Evening",
  "Vehicle_Type": "Scooter",
  "Courier_Experience_yrs": 3.0
}
```

Response:

```json
{"predicted_delivery_time_min": 50.79927433475215}
```

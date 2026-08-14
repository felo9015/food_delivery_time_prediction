# Delivery Time Prediction API

A small FastAPI service wrapping the trained delivery-time model. For how it works internally, see `api/api.md`.

## Setup

From the project root, with the virtual environment active:

```bash
source pandg/bin/activate
pip install -r requirements.txt
```

This installs `fastapi`, `uvicorn`, and `pydantic` along with the rest of the project's dependencies.

## Running the server

From the project root:

```bash
uvicorn api.main:app --reload
```

- `api.main:app` points `uvicorn` (the server) at the `app` object defined in `api/main.py`.
- `--reload` restarts the server automatically whenever the code changes — useful during development, not meant for production.

The server starts at `http://127.0.0.1:8000`. On startup, it loads the trained model once from `model_pipeline/artifacts/model.joblib`.

## Trying it out

FastAPI generates interactive API documentation automatically. With the server running, open:

```
http://127.0.0.1:8000/docs
```

This page lists the `/predict` endpoint, shows the exact fields it expects (including the valid values for `Weather`, `Traffic_Level`, `Time_of_Day`, and `Vehicle_Type` as dropdowns), and lets you send a real request and see the response, directly from the browser — no separate tool needed.

## Example request/response

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Distance_km": 7.5,
    "Preparation_Time_min": 15,
    "Weather": "Rainy",
    "Traffic_Level": "Medium",
    "Time_of_Day": "Evening",
    "Vehicle_Type": "Scooter",
    "Courier_Experience_yrs": 3.0
  }'
```

Response:

```json
{"predicted_delivery_time_min": 50.79927433475215}
```

`Courier_Experience_yrs` can be omitted entirely if unknown — the model imputes it with the median learned during training. `Weather`, `Traffic_Level`, and `Time_of_Day` accept `"Unknown"` as an explicit, valid value for the same reason.

## Running with Docker

```bash
docker build -t delivery-time-api .
docker run -p 8000:8000 delivery-time-api
```

The API is then available at `http://127.0.0.1:8000`, same as running it directly.

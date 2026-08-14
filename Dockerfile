# Minimal image to run the prediction API reproducibly. No CI/CD or
# orchestration here -- just packaging the service itself.
FROM python:3.12-slim

WORKDIR /app

# Installed before copying the rest of the code, so this layer is only
# rebuilt when requirements.txt actually changes, not on every code edit.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api/ api/
COPY model_pipeline/ model_pipeline/

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

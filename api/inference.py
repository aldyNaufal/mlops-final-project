from pathlib import Path
import os, time
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from starlette.responses import Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from prometheus_exporter import (
    INFERENCE_REQUEST_TOTAL,
    INFERENCE_LATENCY_SECONDS,
    INFERENCE_IN_PROGRESS,
    INFERENCE_PREDICTION_PER_GENRE,
    INFERENCE_REQUEST_SIZE,
    INFERENCE_LAST_CONFIDENCE,
    INFERENCE_LAST_PREDICTION_TS,
    INFERENCE_ERROR_TOTAL,
    INFERENCE_QUEUE_LENGTH,
    INFERENCE_MODEL_VERSION,
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path("/app/models") / "tfidf_svc_genre_game_best_tuned_local.pkl"

model = None
if MODEL_PATH.exists():
    model = joblib.load(MODEL_PATH)

INFERENCE_MODEL_VERSION.labels(version="1.0").set(1)

app = FastAPI(title="Genre Game Inference Service", version="1.0.0")

class InferenceRequest(BaseModel):
    title: str
    description: str | None = ""
    tags: str | None = ""

class InferenceResponse(BaseModel):
    predicted_genre: str
    model_version: str = "1.0"

def _combine_text(req: InferenceRequest) -> str:
    parts = [req.title]
    if req.description:
        parts.append(req.description)
    if req.tags:
        parts.append(req.tags)
    return " ".join(parts)

@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": bool(model), "model_path": str(MODEL_PATH)}

@app.post("/predict", response_model=InferenceResponse)
def predict(request: InferenceRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model belum tersedia di /app/models")

    start_time = time.time()
    endpoint = "/predict"
    INFERENCE_IN_PROGRESS.inc()
    INFERENCE_QUEUE_LENGTH.set(0)

    try:
        text = _combine_text(request)
        INFERENCE_REQUEST_SIZE.observe(len(text))

        y_pred = model.predict([text])[0]

        latency = time.time() - start_time
        INFERENCE_LATENCY_SECONDS.labels(endpoint=endpoint).observe(latency)
        INFERENCE_REQUEST_TOTAL.labels(endpoint=endpoint, status="success").inc()
        INFERENCE_PREDICTION_PER_GENRE.labels(genre=y_pred).inc()
        INFERENCE_LAST_CONFIDENCE.set(1.0)
        INFERENCE_LAST_PREDICTION_TS.set(time.time())

        return InferenceResponse(predicted_genre=y_pred, model_version="1.0")

    except Exception as e:
        INFERENCE_REQUEST_TOTAL.labels(endpoint=endpoint, status="error").inc()
        INFERENCE_ERROR_TOTAL.labels(type=e.__class__.__name__).inc()
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        INFERENCE_IN_PROGRESS.dec()

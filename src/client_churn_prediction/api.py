"""Optional FastAPI service for trained models."""

from __future__ import annotations

from pathlib import Path

from .config import load_config, resolve_project_path
from .data import normalize_columns
from .features import model_matrix, prepare_features


try:
    from fastapi import FastAPI
    from pydantic import BaseModel
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("Install API extras with `pip install -r requirements-api.txt`.") from exc


class RecordsPayload(BaseModel):
    records: list[dict]


app = FastAPI(title="Client Churn Prediction API")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: RecordsPayload) -> dict:
    import joblib
    import pandas as pd

    config = load_config()
    model_path = resolve_project_path(config, config.get("modeling", {}).get("model_file", "models/model.joblib"))
    model = joblib.load(model_path)
    df = normalize_columns(pd.DataFrame(payload.records))
    X, _, _ = model_matrix(prepare_features(df, config, training=False), config, training=False)
    predictions = model.predict(X).tolist()
    response = {"prediction": predictions}
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        response["score"] = proba[:, -1].tolist() if proba.ndim == 2 else proba.tolist()
    return response

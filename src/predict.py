"""Serving layer: load the serialized pipeline and expose the churn-scoring contract."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src import config

logger = logging.getLogger(__name__)


class Predictor:
    def __init__(self, artifact_path: Path = config.PIPELINE_PATH) -> None:
        import joblib

        if not Path(artifact_path).exists():
            raise FileNotFoundError(f"No artifact at {artifact_path}. Run `python -m src.pipeline` first.")
        art = joblib.load(artifact_path)
        self.pipeline = art["pipeline"]
        self.base_rate: float = art["base_rate"]
        self.importances: List[Dict[str, Any]] = art.get("importances", [])
        self.best_model: str = art.get("best_model", "")

    def score(self, records: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(records)[:, 1]

    def score_one(self, features: Dict[str, Any]) -> float:
        return float(self.score(pd.DataFrame([features]))[0])

    def rank_percentile(self, score: float, reference_scores: np.ndarray) -> float:
        if len(reference_scores) == 0:
            return 0.0
        return float((reference_scores < score).mean() * 100)

    def top_features(self, n: int = 6) -> List[Dict[str, Any]]:
        return self.importances[:n]

"""Smoke tests for the data contract, leakage guarantees and the serving surface."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src.predict import Predictor  # noqa: E402
from src.preprocessing import FeaturePrep, Preprocessor, engineer  # noqa: E402


@pytest.fixture(scope="module")
def sample():
    return pd.read_csv(config.SAMPLE_PATH)


def test_target_present_and_imbalanced(sample):
    assert config.TARGET in sample.columns
    assert 0.10 < sample[config.TARGET].mean() < 0.35


def test_id_columns_absent_from_features(sample):
    X, y = Preprocessor().run(sample)
    for col in config.ID_COLS:
        assert col not in X.columns
    assert set(y.unique()) <= {0, 1}


def test_engineered_features_present(sample):
    eng = engineer(sample.head(50))
    for col in config.ENGINEERED_FEATURES:
        assert col in eng.columns


def test_feature_prep_fixed_columns(sample):
    a = FeaturePrep().fit_transform(sample.head(20))
    b = FeaturePrep().transform(sample.head(5))
    assert list(a.columns) == list(b.columns)


def test_predictor_contract():
    pred = Predictor()
    rec = {"CreditScore": 600, "Geography": "Germany", "Gender": "Female", "Age": 50,
           "Tenure": 2, "Balance": 120000.0, "NumOfProducts": 1, "HasCrCard": 1,
           "IsActiveMember": 0, "EstimatedSalary": 50000.0}
    s = pred.score_one(rec)
    assert 0.0 <= s <= 1.0
    assert 0.0 < pred.base_rate < 1.0
    assert len(pred.top_features(5)) >= 1


def test_higher_risk_scores_above_lower_risk():
    pred = Predictor()
    risky = {"CreditScore": 500, "Geography": "Germany", "Gender": "Female", "Age": 55,
             "Tenure": 1, "Balance": 150000.0, "NumOfProducts": 1, "HasCrCard": 0,
             "IsActiveMember": 0, "EstimatedSalary": 40000.0}
    safe = {**risky, "Age": 30, "IsActiveMember": 1, "NumOfProducts": 2, "Geography": "France", "Balance": 0.0}
    assert pred.score_one(risky) > pred.score_one(safe)

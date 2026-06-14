"""Transformation layer. Feature engineering (balance/salary ratio, zero-balance
flag, products-per-tenure) lives in a custom first pipeline step so training and
serving share the identical transform.
"""
from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config

logger = logging.getLogger(__name__)

SOURCE_COLUMNS = list(config.NUMERIC_FEATURES) + ["HasCrCard", "IsActiveMember", "Gender", "Geography"]
NUMERIC_MODEL = list(config.NUMERIC_FEATURES) + ["HasCrCard", "IsActiveMember", "Gender"] + list(config.ENGINEERED_FEATURES)


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Gender" in out:
        out["Gender"] = out["Gender"].map({"Male": 1, "Female": 0}).fillna(out["Gender"]).astype(float)
    out["BalanceSalaryRatio"] = out["Balance"] / (out["EstimatedSalary"].abs() + 1)
    out["ZeroBalance"] = (out["Balance"] == 0).astype(float)
    out["ProductsPerTenure"] = out["NumOfProducts"] / (out["Tenure"] + 1)
    return out


class FeaturePrep(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X) -> pd.DataFrame:
        df = engineer(pd.DataFrame(X).copy())
        cols = NUMERIC_MODEL + ["Geography"]
        for c in cols:
            if c not in df.columns:
                df[c] = np.nan if c in NUMERIC_MODEL else "France"
        return df[cols]


def build_column_transformer() -> ColumnTransformer:
    numeric_pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical_pipe = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                                 ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer(
        [("num", numeric_pipe, NUMERIC_MODEL), ("cat", categorical_pipe, ["Geography"])],
        remainder="drop")


class Preprocessor:
    def __init__(self, processed_path=config.PROCESSED_PATH) -> None:
        self.processed_path = processed_path

    def run(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        if config.TARGET not in df.columns:
            raise ValueError(f"Target '{config.TARGET}' missing")
        y = df[config.TARGET].astype(int)
        X = df[[c for c in SOURCE_COLUMNS if c in df.columns]].copy()
        self.processed_path.parent.mkdir(parents=True, exist_ok=True)
        cleaned = engineer(df).copy()
        cleaned[config.TARGET] = y.values
        cleaned.to_parquet(self.processed_path, index=False)
        logger.info("Processed frame (%d rows) written to %s", len(cleaned), self.processed_path)
        return X, y

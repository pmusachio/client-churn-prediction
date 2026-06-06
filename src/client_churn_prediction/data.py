"""Data loading and lightweight profiling utilities."""

from __future__ import annotations

import json
import re
from pathlib import Path

from .config import load_config, resolve_project_path


def snake_case(value: str) -> str:
    value = str(value).strip()
    value = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", value)
    value = re.sub(r"[^0-9A-Za-z]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_").lower()
    return value or "unnamed"


def normalize_columns(df):
    """Normalize column names to snake_case and handle unnamed index columns."""
    columns = []
    seen = {}
    for column in df.columns:
        name = snake_case(column)
        if name.startswith("unnamed"):
            name = "row_id"
        count = seen.get(name, 0)
        seen[name] = count + 1
        columns.append(name if count == 0 else f"{name}_{count}")
    df = df.copy()
    df.columns = columns
    return df


def read_csv(path: str | Path, **kwargs):
    import csv
    import pandas as pd

    path = Path(path)
    detected = {"encoding": "utf-8", "sep": ","}
    for encoding in ["utf-8", "latin1"]:
        try:
            sample = path.read_text(encoding=encoding, errors="strict")[:32768]
            dialect = csv.Sniffer().sniff(sample, delimiters=",;	|")
            detected = {"encoding": encoding, "sep": dialect.delimiter}
            break
        except UnicodeDecodeError:
            continue
        except csv.Error:
            detected = {"encoding": encoding, "sep": ","}
            break
    options = {"low_memory": False, **detected, **kwargs}
    return pd.read_csv(path, **options)


def configured_path(config: dict, key: str) -> Path | None:
    return resolve_project_path(config, config.get("data", {}).get(key))


def load_training_frame(config: dict | None = None):
    config = config or load_config()
    train_path = configured_path(config, "train_file")
    if not train_path or not train_path.exists():
        raise FileNotFoundError(
            f"Training file not found. Configure [data].train_file or place data at {train_path}."
        )

    df = read_csv(train_path)
    store_path = configured_path(config, "store_file")
    if store_path and store_path.exists():
        store = read_csv(store_path)
        join_key = config.get("data", {}).get("join_key", "store")
        df = normalize_columns(df).merge(normalize_columns(store), how="left", on=join_key)
    return normalize_columns(df)


def load_test_frame(config: dict | None = None):
    config = config or load_config()
    test_path = configured_path(config, "test_file")
    if not test_path or not test_path.exists():
        return None
    df = read_csv(test_path)
    store_path = configured_path(config, "store_file")
    if store_path and store_path.exists():
        store = read_csv(store_path)
        join_key = config.get("data", {}).get("join_key", "store")
        df = normalize_columns(df).merge(normalize_columns(store), how="left", on=join_key)
    return normalize_columns(df)


def profile_dataframe(df, sample_rows: int = 5) -> dict:
    missing = df.isna().mean().sort_values(ascending=False)
    return {
        "shape": list(df.shape),
        "columns": df.columns.tolist(),
        "dtypes": {column: str(dtype) for column, dtype in df.dtypes.items()},
        "missing_share": {column: float(value) for column, value in missing.items()},
        "sample": df.head(sample_rows).to_dict(orient="records"),
    }


def write_profile(config_path: str | Path | None = None) -> Path:
    config = load_config(config_path)
    df = load_training_frame(config)
    output = resolve_project_path(config, "reports/data_profile.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(profile_dataframe(df), indent=2, default=str), encoding="utf-8")
    return output

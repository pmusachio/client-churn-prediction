"""Send a sample record to the local FastAPI /predict endpoint."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys
from urllib import request


def package_name(root: Path) -> str:
    src = root / "src"
    for child in src.iterdir():
        if child.is_dir() and (child / "config.py").exists():
            return child.name
    raise RuntimeError("Could not find package under src/.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000/predict")
    parser.add_argument("--input", default=None, help="Optional CSV to sample from.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    pkg = package_name(root)
    config_mod = importlib.import_module(f"{pkg}.config")
    data_mod = importlib.import_module(f"{pkg}.data")

    config = config_mod.load_config(root / "configs" / "project.toml")
    data_cfg = config.get("data", {})
    input_path = Path(args.input) if args.input else root / data_cfg.get("test_file", data_cfg.get("train_file", ""))
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = data_mod.read_csv(input_path).head(1)
    target = data_cfg.get("target")
    if target in df.columns:
        df = df.drop(columns=[target])
    payload = {"records": df.to_dict(orient="records")}
    body = json.dumps(payload, default=str).encode("utf-8")
    req = request.Request(args.url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    with request.urlopen(req, timeout=30) as response:
        print(response.read().decode("utf-8"))


if __name__ == "__main__":
    main()

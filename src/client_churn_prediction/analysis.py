"""Project-specific analytical routines for non-standard ML tasks."""

from __future__ import annotations

import json
import math
from pathlib import Path

from .config import load_config, resolve_project_path
from .data import load_training_frame, normalize_columns


def _normal_two_sided_pvalue(z: float) -> float:
    return math.erfc(abs(z) / math.sqrt(2))


def analyze_ab_test(config: dict) -> dict:
    df = normalize_columns(load_training_frame(config))
    data = config.get("data", {})
    group_col = data.get("group_column", "group")
    metric_col = data.get("metric_column", "converted")
    control = data.get("control_value", "control")
    treatment = data.get("treatment_value", "treatment")

    if group_col not in df.columns:
        raise ValueError(f"Missing group column: {group_col}")
    if metric_col not in df.columns:
        if "purchases" in df.columns:
            metric_col = "purchases"
        else:
            raise ValueError(f"Missing metric column: {metric_col}")

    summary = df.groupby(group_col)[metric_col].agg(["count", "mean", "std", "sum"]).reset_index()
    rates = dict(zip(summary[group_col], summary["mean"]))
    counts = dict(zip(summary[group_col], summary["count"]))

    if control not in rates or treatment not in rates:
        groups = list(rates)
        control, treatment = groups[0], groups[-1]

    p1, p2 = float(rates[control]), float(rates[treatment])
    n1, n2 = int(counts[control]), int(counts[treatment])
    pooled = (p1 * n1 + p2 * n2) / (n1 + n2)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2)) if n1 and n2 else float("nan")
    z = (p2 - p1) / se if se else float("nan")
    result = {
        "control": control,
        "treatment": treatment,
        "control_rate": p1,
        "treatment_rate": p2,
        "absolute_lift": p2 - p1,
        "relative_lift": (p2 - p1) / p1 if p1 else None,
        "z_stat": z,
        "p_value": _normal_two_sided_pvalue(z) if not math.isnan(z) else None,
        "summary": summary.to_dict(orient="records"),
    }
    output = resolve_project_path(config, "reports/ab_test_results.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return {"results_path": str(output), "results": result}


def analyze_price_elasticity(config: dict) -> dict:
    import numpy as np
    import pandas as pd

    df = normalize_columns(load_training_frame(config))
    data = config.get("data", {})
    product_col = data.get("product_column", "name")
    price_col = data.get("price_column", "disc_price")
    demand_col = data.get("demand_column", "imp_count")
    min_observations = int(config.get("modeling", {}).get("min_observations", 12))

    for column in [price_col, demand_col]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df[(df[price_col] > 0) & (df[demand_col] > 0)].dropna(subset=[product_col, price_col, demand_col])

    rows = []
    for product, group in df.groupby(product_col):
        if len(group) < min_observations or group[price_col].nunique() < 2:
            continue
        x = np.log(group[price_col].to_numpy())
        y = np.log(group[demand_col].to_numpy())
        slope, intercept = np.polyfit(x, y, deg=1)
        y_hat = intercept + slope * x
        ss_res = float(((y - y_hat) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        r2 = 1 - ss_res / ss_tot if ss_tot else 0.0
        rows.append(
            {
                "product": product,
                "observations": int(len(group)),
                "price_elasticity": float(slope),
                "r2": float(r2),
                "avg_price": float(group[price_col].mean()),
                "avg_demand": float(group[demand_col].mean()),
            }
        )

    result = pd.DataFrame(rows).sort_values("price_elasticity")
    output = resolve_project_path(config, "reports/price_elasticity.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    return {"results_path": str(output), "n_products": int(len(result))}


def run_analysis(config_path: str | Path | None = None):
    config = load_config(config_path)
    problem_type = config.get("project", {}).get("problem_type")
    if problem_type == "ab_testing":
        return analyze_ab_test(config)
    if problem_type == "price_elasticity":
        return analyze_price_elasticity(config)
    raise ValueError(f"No special analysis routine for problem_type={problem_type!r}. Use train instead.")

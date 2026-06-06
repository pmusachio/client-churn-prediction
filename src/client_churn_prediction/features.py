"""Feature engineering utilities — churn-domain and generic project workflows."""

from __future__ import annotations

import math

from .data import normalize_columns


def target_name(config: dict) -> str | None:
    target = config.get("data", {}).get("target")
    if not target:
        return None
    return _normalize_name(target)


def _normalize_name(name: str) -> str:
    import pandas as pd

    return normalize_columns(pd.DataFrame(columns=[name])).columns[0]


def _safe_datetime(series, fmt: str | None = None):
    import pandas as pd

    return pd.to_datetime(series, format=fmt, errors="coerce")


def _add_date_parts(df, column: str, prefix: str | None = None, fmt: str | None = None):
    if column not in df.columns:
        return df
    prefix = prefix or column
    value = _safe_datetime(df[column], fmt)
    df[f"{prefix}_year"] = value.dt.year
    df[f"{prefix}_month"] = value.dt.month
    df[f"{prefix}_day"] = value.dt.day
    df[f"{prefix}_dayofweek"] = value.dt.dayofweek
    return df


# ---------------------------------------------------------------------------
# Churn-domain features (Géron Ch.2: domain-driven feature engineering)
# ---------------------------------------------------------------------------

def _churn_features(df):
    """
    Create domain-informed features for bank churn prediction.

    Inspired by the feature engineering principles in Aurélien Géron's
    *Hands-On Machine Learning with Scikit-Learn and PyTorch* (2025),
    Chapter 2: combining existing attributes to produce more useful signals.
    """
    # --- balance-based signals ---
    if "balance" in df.columns:
        df["is_zero_balance"] = (df["balance"] == 0).astype(int)

    if {"balance", "num_of_products"}.issubset(df.columns):
        df["balance_per_product"] = df["balance"] / (df["num_of_products"].replace(0, 1))

    if {"balance", "estimated_salary"}.issubset(df.columns):
        df["balance_salary_ratio"] = df["balance"] / (df["estimated_salary"].replace(0, 1))

    # --- age / tenure signals ---
    if {"age", "tenure"}.issubset(df.columns):
        df["age_tenure_ratio"] = df["age"] / (df["tenure"] + 1)
        df["is_senior"] = (df["age"] >= 50).astype(int)

    # --- engagement signals ---
    if {"num_of_products", "is_active_member"}.issubset(df.columns):
        df["products_active_combo"] = df["num_of_products"] * df["is_active_member"]

    if {"age", "is_active_member"}.issubset(df.columns):
        df["senior_inactive"] = ((df["age"] >= 50) & (df["is_active_member"] == 0)).astype(int)

    if {"num_of_products", "has_cr_card"}.issubset(df.columns):
        df["multi_product_with_card"] = (
            (df["num_of_products"] > 1) & (df["has_cr_card"] == 1)
        ).astype(int)

    # --- credit-score segments ---
    if "credit_score" in df.columns:
        df["credit_score_band"] = df["credit_score"].apply(_credit_band)

    return df


def _credit_band(score) -> str:
    if score < 580:
        return "poor"
    if score < 670:
        return "fair"
    if score < 740:
        return "good"
    if score < 800:
        return "very_good"
    return "exceptional"


# ---------------------------------------------------------------------------
# Other dataset families (kept for multi-project template compatibility)
# ---------------------------------------------------------------------------

def _airbnb_features(df):
    df = _add_date_parts(df, "date_account_created", "account_created")
    if "timestamp_first_active" in df.columns:
        df["timestamp_first_active"] = df["timestamp_first_active"].astype(str).str.replace(r"\.0$", "", regex=True)
        df = _add_date_parts(df, "timestamp_first_active", "first_active", "%Y%m%d%H%M%S")
    if "age" in df.columns:
        current_year = 2026
        birth_year = df["age"].between(1900, current_year)
        df.loc[birth_year, "age"] = current_year - df.loc[birth_year, "age"]
        df.loc[(df["age"] < 14) | (df["age"] > 100), "age"] = None
    for column in ["gender", "first_affiliate_tracked"]:
        if column in df.columns:
            df[column] = df[column].replace({"-unknown-": None, "unknown": None})
    return df


def _fraud_features(df):
    if "step" in df.columns:
        df["transaction_day"] = (df["step"] // 24 + 1).astype("Int64")
        df["is_weekend_cycle"] = (df["transaction_day"] % 7).isin([0, 6]).astype(int)
    if {"new_balance_orig", "old_balance_orig"}.issubset(df.columns):
        df["origin_balance_delta"] = df["new_balance_orig"] - df["old_balance_orig"]
    if {"new_balance_dest", "old_balance_dest"}.issubset(df.columns):
        df["destination_balance_delta"] = df["new_balance_dest"] - df["old_balance_dest"]
    if {"amount", "old_balance_orig"}.issubset(df.columns):
        df["amount_to_origin_balance"] = df["amount"] / (df["old_balance_orig"].abs() + 1)
    return df


def _health_features(df):
    if "vehicle_age" in df.columns:
        df["vehicle_age"] = (
            df["vehicle_age"]
            .astype(str)
            .str.lower()
            .str.replace("> 2 years", "over_2_years", regex=False)
            .str.replace("1-2 year", "between_1_2_years", regex=False)
            .str.replace("< 1 year", "below_1_year", regex=False)
        )
    if "vehicle_damage" in df.columns:
        df["vehicle_damage"] = df["vehicle_damage"].map({"Yes": 1, "No": 0, "yes": 1, "no": 0}).fillna(df["vehicle_damage"])
    if "gender" in df.columns:
        df["gender"] = df["gender"].map({"Male": 1, "Female": 0, "male": 1, "female": 0}).fillna(df["gender"])
    return df


def _cardio_features(df):
    if "age" in df.columns and "age_years" not in df.columns:
        df["age_years"] = (df["age"] / 365.25).round(1)
    if {"height", "weight"}.issubset(df.columns):
        height_m = df["height"] / 100
        df["bmi"] = df["weight"] / (height_m * height_m)
        df.loc[(df["bmi"] < 10) | (df["bmi"] > 80), "bmi"] = None
    if {"ap_hi", "ap_lo"}.issubset(df.columns):
        df["pulse_pressure"] = df["ap_hi"] - df["ap_lo"]
        df["high_blood_pressure"] = ((df["ap_hi"] >= 140) | (df["ap_lo"] >= 90)).astype(int)
    return df


def _rossmann_features(df):
    if "date" in df.columns:
        value = _safe_datetime(df["date"])
        df["year"] = value.dt.year
        df["month"] = value.dt.month
        df["day"] = value.dt.day
        df["week_of_year"] = value.dt.isocalendar().week.astype("Int64")
        df["day_of_year"] = value.dt.dayofyear
        df["month_sin"] = (2 * math.pi * df["month"] / 12).apply(math.sin)
        df["month_cos"] = (2 * math.pi * df["month"] / 12).apply(math.cos)
        if "day_of_week" in df.columns:
            df["day_of_week_sin"] = (2 * math.pi * df["day_of_week"] / 7).apply(math.sin)
            df["day_of_week_cos"] = (2 * math.pi * df["day_of_week"] / 7).apply(math.cos)
    if "competition_distance" in df.columns:
        df["competition_distance"] = df["competition_distance"].fillna(df["competition_distance"].median())
    if "state_holiday" in df.columns:
        df["state_holiday"] = df["state_holiday"].replace({"0": "regular_day", 0: "regular_day"})
    return df


def _price_features(df):
    for column in ["price", "disc_price", "imp_count"]:
        if column in df.columns:
            df[column] = df[column].astype(str).str.replace(",", ".", regex=False)
            df[column] = df[column].str.extract(r"([-+]?[0-9]*\.?[0-9]+)", expand=False)
            df[column] = df[column].astype(float)
    if "date_imp_d" in df.columns:
        df = _add_date_parts(df, "date_imp_d", "observed")
    if {"price", "disc_price"}.issubset(df.columns):
        df["discount_rate"] = 1 - (df["disc_price"] / df["price"].replace(0, float("nan")))
    return df


def build_rfm_features(df, config: dict):
    import pandas as pd

    data = config.get("data", {})
    customer_col = _normalize_name(data.get("customer_column", "CustomerID"))
    invoice_col = _normalize_name(data.get("invoice_column", "InvoiceNo"))
    date_col = _normalize_name(data.get("date_column", "InvoiceDate"))
    quantity_col = _normalize_name(data.get("quantity_column", "Quantity"))
    price_col = _normalize_name(data.get("price_column", "UnitPrice"))

    df = normalize_columns(df)
    required = {customer_col, invoice_col, date_col, quantity_col, price_col}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing RFM columns: {sorted(missing)}")

    df = df.dropna(subset=[customer_col]).copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df[df[date_col].notna()]
    df = df[(df[quantity_col] > 0) & (df[price_col] > 0)]
    if invoice_col in df.columns:
        df = df[~df[invoice_col].astype(str).str.startswith("C", na=False)]
    df["gross_revenue"] = df[quantity_col] * df[price_col]
    reference_date = df[date_col].max() + pd.Timedelta(days=1)
    grouped = df.groupby(customer_col).agg(
        recency_days=(date_col, lambda x: (reference_date - x.max()).days),
        frequency=(invoice_col, "nunique"),
        monetary=("gross_revenue", "sum"),
        avg_ticket=("gross_revenue", "mean"),
        total_items=(quantity_col, "sum"),
    )
    return grouped.reset_index().rename(columns={customer_col: "customer_id"})


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def prepare_features(df, config: dict, training: bool = True):
    df = normalize_columns(df)
    family = config.get("project", {}).get("family", "generic")
    if family == "churn":
        df = _churn_features(df)
    elif family == "airbnb":
        df = _airbnb_features(df)
    elif family == "fraud":
        df = _fraud_features(df)
    elif family == "health_cross_sell":
        df = _health_features(df)
    elif family == "cardio":
        df = _cardio_features(df)
    elif family == "rossmann":
        df = _rossmann_features(df)
    elif family == "price_elasticity":
        df = _price_features(df)
    elif family == "rfm_clustering":
        return build_rfm_features(df, config)
    return df


def model_matrix(df, config: dict, training: bool = True):
    df = prepare_features(df, config, training=training)
    target = _normalize_name(config.get("data", {}).get("target", "")) if config.get("data", {}).get("target") else None
    drop_columns = {_normalize_name(c) for c in config.get("data", {}).get("drop_columns", [])}
    id_columns = {_normalize_name(c) for c in config.get("data", {}).get("id_columns", [])}

    y = None
    if target and target in df.columns:
        y = df[target]
        drop_columns.add(target)

    X = df.drop(columns=[c for c in drop_columns.union(id_columns) if c in df.columns], errors="ignore")
    return X, y, df

"""Interactive churn propensity dashboard.

Scores a customer's churn probability and shows where they fall in the retention
priority list, plus a campaign view (cumulative gains and lift at a chosen capacity)
computed on the versioned sample.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.predict import Predictor  # noqa: E402

D = config.DRACULA
st.set_page_config(page_title="Churn Propensity Ranking", layout="wide")
st.markdown(
    f"""<style>
    .stApp {{ background-color: {D['background']}; color: {D['foreground']}; }}
    section[data-testid="stSidebar"] {{ background-color: {D['current_line']}; }}
    h1, h2, h3 {{ color: {D['purple']}; }}
    </style>""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_predictor() -> Predictor:
    return Predictor()


@st.cache_data
def load_sample() -> pd.DataFrame:
    return pd.read_csv(config.SAMPLE_PATH) if config.SAMPLE_PATH.exists() else pd.DataFrame()


@st.cache_data
def sample_scores() -> np.ndarray:
    df = load_sample()
    return load_predictor().score(df) if not df.empty else np.array([])


def style_axes(ax):
    ax.set_facecolor(D["background"])
    for s in ax.spines.values():
        s.set_color(D["current_line"])
    ax.tick_params(colors=D["foreground"])
    ax.xaxis.label.set_color(D["foreground"])
    ax.yaxis.label.set_color(D["foreground"])
    ax.grid(True, color=D["current_line"], linestyle="--", alpha=0.4)


def gains_chart(scores, y, capacity_pct):
    order = np.argsort(scores)[::-1]
    ys = y[order]
    cum = np.cumsum(ys) / ys.sum()
    pct = np.arange(1, len(ys) + 1) / len(ys)
    fig, ax = plt.subplots(figsize=(6, 3.4), facecolor=D["background"])
    ax.plot(pct * 100, cum * 100, color=D["green"], linewidth=2, label="Model")
    ax.plot([0, 100], [0, 100], color=D["comment"], linestyle="--", linewidth=1.5, label="Random")
    ax.axvline(capacity_pct, color=D["pink"], linestyle=":", linewidth=1.5)
    ax.set_xlabel("Customers targeted (%)")
    ax.set_ylabel("Churners captured (%)")
    ax.legend(facecolor=D["current_line"], edgecolor=D["comment"], labelcolor=D["foreground"], fontsize=8)
    style_axes(ax)
    fig.tight_layout()
    return fig


def main():
    try:
        predictor = load_predictor()
    except FileNotFoundError:
        st.error("Model artifact not found. Run the pipeline before launching the app.")
        return

    st.title("Client Churn Prediction — Retention Ranking")
    st.markdown(
        "Scores a bank customer's churn risk so a limited retention budget targets the customers "
        "most likely to leave."
    )

    with st.sidebar:
        st.header("Customer")
        age = st.slider("Age", 18, 90, 45)
        geography = st.selectbox("Geography", list(config.GEOGRAPHIES), index=2)
        gender = st.selectbox("Gender", ["Female", "Male"])
        credit = st.slider("Credit score", 350, 850, 600)
        tenure = st.slider("Tenure (years)", 0, 10, 2)
        balance = st.number_input("Balance", 0.0, 250000.0, 120000.0, 1000.0)
        products = st.selectbox("Number of products", [1, 2, 3, 4], index=0)
        active = st.selectbox("Active member", [1, 0], format_func=lambda v: "Yes" if v else "No", index=1)
        has_card = st.selectbox("Has credit card", [1, 0], format_func=lambda v: "Yes" if v else "No")
        salary = st.number_input("Estimated salary", 0.0, 200000.0, 50000.0, 1000.0)
        run = st.button("Score customer", type="primary")

    record = {"CreditScore": credit, "Geography": geography, "Gender": gender, "Age": age,
              "Tenure": tenure, "Balance": balance, "NumOfProducts": products, "HasCrCard": has_card,
              "IsActiveMember": active, "EstimatedSalary": salary}

    if run:
        score = predictor.score_one(record)
        pct = predictor.rank_percentile(score, sample_scores())
        st.subheader("Churn risk")
        c = st.columns(3)
        c[0].metric("Churn probability", f"{score*100:.1f}%")
        c[1].metric("Retention priority (top)", f"{100 - pct:.0f}%")
        c[2].metric("Base churn rate", f"{predictor.base_rate*100:.1f}%")
        verdict = "high priority" if pct >= 80 else "medium priority" if pct >= 50 else "low priority"
        st.markdown(
            f"This customer is more at risk than {pct:.0f}% of the base, a **{verdict}** retention target "
            f"({score/predictor.base_rate:.1f}x the average churn risk)."
        )
        st.subheader("Most influential features (model-wide)")
        imp = pd.DataFrame(predictor.top_features(6)).rename(
            columns={"feature": "Feature", "importance": "Permutation importance (ROC AUC drop)"})
        st.dataframe(imp, hide_index=True, width="stretch")

    df = load_sample()
    if not df.empty and config.TARGET in df.columns:
        st.subheader("Retention campaign view (reference sample)")
        capacity = st.slider("Targeting capacity (% of base)", 5, 100, config.DEFAULT_CAPACITY_PCT, 5)
        scores = sample_scores()
        y = df[config.TARGET].to_numpy()
        order = np.argsort(scores)[::-1]
        k = int(len(y) * capacity / 100)
        captured = y[order][:k].sum() / y.sum() if y.sum() else 0
        lift = (y[order][:k].mean() / y.mean()) if (k and y.mean()) else 0
        left, right = st.columns([2, 1])
        with left:
            st.pyplot(gains_chart(scores, y, capacity))
        with right:
            st.metric("Churners captured", f"{captured*100:.0f}%")
            st.metric("Lift vs random", f"{lift:.2f}x")
            st.caption(f"Targeting the top {capacity}% reaches {captured*100:.0f}% of churners.")


if __name__ == "__main__":
    main()

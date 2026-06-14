"""Central configuration: paths, dataset identity, modeling constants and the
Dracula palette shared by the pipeline, the serving layer and the dashboard.
"""
from __future__ import annotations

from pathlib import Path

BASE_DIR: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = BASE_DIR / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
SAMPLE_DIR: Path = DATA_DIR / "sample"
MODELS_DIR: Path = BASE_DIR / "models"

PIPELINE_PATH: Path = MODELS_DIR / "pipeline.joblib"
MODEL_CARD_PATH: Path = MODELS_DIR / "model_card.json"
PROCESSED_PATH: Path = PROCESSED_DIR / "train.parquet"

SAMPLE_FILENAME: str = "churn_sample.csv"
SAMPLE_PATH: Path = SAMPLE_DIR / SAMPLE_FILENAME

KAGGLE_DATASET: str = "mervetorkan/churndataset"
RAW_FILENAME: str = "churn.csv"

TARGET: str = "Exited"
POSITIVE_LABEL: int = 1
ID_COLS: tuple[str, ...] = ("RowNumber", "CustomerId", "Surname")
# No target leakage: all attributes are observed while the customer is still active.

NUMERIC_FEATURES: tuple[str, ...] = (
    "CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "EstimatedSalary",
)
BINARY_FEATURES: tuple[str, ...] = ("HasCrCard", "IsActiveMember", "Gender")  # Gender mapped 0/1
CATEGORICAL_FEATURES: tuple[str, ...] = ("Geography",)
ENGINEERED_FEATURES: tuple[str, ...] = ("BalanceSalaryRatio", "ZeroBalance", "ProductsPerTenure")

GEOGRAPHIES: tuple[str, ...] = ("France", "Spain", "Germany")

TEST_SIZE: float = 0.2
SEED: int = 42
CV_FOLDS: int = 5
TUNING_ITERS: int = 20
SCORING: str = "roc_auc"

CONTACT_CAPACITIES: tuple[int, ...] = (200, 500, 1000)
DEFAULT_CAPACITY_PCT: int = 20

DRACULA = {
    "background": "#282a36", "current_line": "#44475a", "foreground": "#f8f8f2",
    "comment": "#6272a4", "cyan": "#8be9fd", "green": "#50fa7b", "orange": "#ffb86c",
    "pink": "#ff79c6", "purple": "#bd93f9", "red": "#ff5555", "yellow": "#f1fa8c",
}

"""Acquisition layer: pull the churn dataset from Kaggle on demand, with the
versioned sample as an offline fallback.
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

import pandas as pd

from src import config

logger = logging.getLogger(__name__)


class DataLoader:
    def __init__(self, raw_dir: Path = config.RAW_DIR, sample_path: Path = config.SAMPLE_PATH,
                 dataset: str = config.KAGGLE_DATASET) -> None:
        self.raw_dir = raw_dir
        self.sample_path = sample_path
        self.dataset = dataset
        self.raw_path = raw_dir / config.RAW_FILENAME

    def download(self, force: bool = False) -> Path:
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        if self.raw_path.exists() and not force:
            logger.info("Raw file already present at %s", self.raw_path)
            return self.raw_path
        try:
            self._download_from_kaggle()
        except Exception as exc:  # noqa: BLE001 - degrade to the sample
            logger.warning("Kaggle download unavailable (%s); using versioned sample", exc)
            self._copy_from_sample()
        return self.raw_path

    def _download_from_kaggle(self) -> None:
        import kagglehub

        logger.info("Downloading %s from Kaggle", self.dataset)
        cache_dir = Path(kagglehub.dataset_download(self.dataset))
        csvs = sorted(cache_dir.rglob("*.csv"), key=lambda p: p.stat().st_size, reverse=True)
        if not csvs:
            raise FileNotFoundError(f"No CSV in Kaggle download at {cache_dir}")
        shutil.copyfile(csvs[0], self.raw_path)
        logger.info("Raw file written to %s", self.raw_path)

    def _copy_from_sample(self) -> None:
        if not self.sample_path.exists():
            raise FileNotFoundError(f"No Kaggle access and no sample at {self.sample_path}")
        shutil.copyfile(self.sample_path, self.raw_path)
        logger.info("Raw file seeded from sample %s", self.raw_path)

    def load(self) -> pd.DataFrame:
        path = self.raw_path if self.raw_path.exists() else self.download()
        df = pd.read_csv(path)
        logger.info("Loaded %d rows x %d cols from %s", df.shape[0], df.shape[1], path)
        return df

"""Single entrypoint: download -> preprocess -> baseline -> CV select + tune ->
evaluate -> business translation -> importances -> serialize. Idempotent.

    python -m src.pipeline
"""
from __future__ import annotations

import logging

from src import config
from src.data_loader import DataLoader
from src.preprocessing import Preprocessor
from src.train import ModelTrainer


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")


def run() -> None:
    configure_logging()
    log = logging.getLogger("pipeline")
    log.info("Stage 1/6 - acquisition")
    loader = DataLoader()
    raw_path = loader.download()
    df = loader.load()
    log.info("Stage 2/6 - preprocessing")
    X, y = Preprocessor().run(df)
    log.info("Stage 3/6 - baseline")
    trainer = ModelTrainer(X, y, data_source=raw_path)
    trainer.fit_baseline()
    log.info("Stage 4/6 - CV selection + tuning")
    trainer.fit()
    log.info("Stage 5/6 - evaluation, business translation, importances")
    trainer.evaluate()
    trainer.to_business_metrics()
    trainer.compute_importances()
    log.info("Stage 6/6 - serialization")
    trainer.save()
    log.info("Done. Artifact: %s | Card: %s", config.PIPELINE_PATH, config.MODEL_CARD_PATH)


if __name__ == "__main__":
    run()

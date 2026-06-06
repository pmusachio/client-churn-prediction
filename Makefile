.PHONY: setup data profile train compare tune predict api api-test test run-all notebooks clean

VENV    := .venv
PIP     := $(VENV)/bin/pip
PYTHON  := $(VENV)/bin/python
PYTEST  := $(VENV)/bin/pytest
UVICORN := $(VENV)/bin/uvicorn

# ── Environment ──────────────────────────────────────────────────────────────

setup:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt -r requirements-api.txt
	@echo "\n✓ Environment ready. Activate with: source $(VENV)/bin/activate"

# ── Data ─────────────────────────────────────────────────────────────────────

data:
	@mkdir -p data/raw
	@echo "Place data/raw/churn.csv (Kaggle: mervetorkan/churndataset)"
	@echo "Or run:  kaggle datasets download -d mervetorkan/churndataset --unzip -p data/raw"

# ── Pipeline ─────────────────────────────────────────────────────────────────

profile:
	PYTHONPATH=src $(PYTHON) -m client_churn_prediction.cli profile

train:
	PYTHONPATH=src $(PYTHON) -m client_churn_prediction.cli train

compare:
	PYTHONPATH=src $(PYTHON) -c "\
from client_churn_prediction.models import compare_models; \
import json; print(json.dumps(compare_models(), indent=2))"

tune:
	PYTHONPATH=src $(PYTHON) -c "\
from client_churn_prediction.models import tune_model; \
import json; print(json.dumps(tune_model(), indent=2))"

predict:
	PYTHONPATH=src $(PYTHON) -m client_churn_prediction.cli predict \
		--input data/raw/churn.csv --output data/processed/predictions.csv

# ── API ───────────────────────────────────────────────────────────────────────

api:
	PYTHONPATH=src $(UVICORN) client_churn_prediction.api:app --reload --port 8000

api-test:
	$(PYTHON) scripts/sample_api_request.py

# ── Quality ──────────────────────────────────────────────────────────────────

test:
	PYTHONPATH=src $(PYTEST) tests/ -v

# ── Full pipeline (one-shot) ──────────────────────────────────────────────────

run-all: profile train predict
	@echo "\n✓ Full pipeline completed."

# ── Notebooks ─────────────────────────────────────────────────────────────────

notebooks:
	$(VENV)/bin/jupyter lab notebooks/

# ── Cleanup ───────────────────────────────────────────────────────────────────

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache

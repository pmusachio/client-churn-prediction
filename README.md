# Client Churn Prediction

> **End-to-end machine learning project** — rank bank customers by churn probability to guide targeted retention campaigns.

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4%2B-orange)](https://scikit-learn.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-green)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## Business Problem

A retail bank needs to identify which customers are most likely to close their accounts (churn) so the retention team can act proactively. The current approach — contacting customers at random — wastes budget and misses high-risk clients.

**Objective:** Build a scoring model that ranks customers by churn probability, allowing the retention team to focus their limited budget on the top-k highest-risk customers.

**Primary metrics:** ROC AUC · Average Precision · Recall@k · Lift@k

---

## Solution Strategy

This project follows the **8-step end-to-end ML workflow** from Aurélien Géron's *Hands-On Machine Learning with Scikit-Learn and PyTorch* (O'Reilly, 2025):

| Step | Notebook | Description |
|------|----------|-------------|
| 1. Frame the problem | `01_business_understanding` | Business context, metrics, ML framing |
| 2. Get & explore data | `02_data_exploration` | EDA, distributions, correlations, insights |
| 3. Feature engineering | `03_feature_engineering` | Domain features, sklearn transformers |
| 4. Select a model | `04_model_comparison` | 4 models compared via stratified CV |
| 5. Fine-tune | `05_hyperparameter_tuning` | RandomizedSearchCV, feature importance, ROI |
| 6. Deploy | `06_deployment` | FastAPI endpoint, batch scoring, cloud deploy |

---

## Dataset

Source: [Kaggle — Churn Modelling Dataset](https://www.kaggle.com/datasets/mervetorkan/churndataset)

| Property | Value |
|---|---|
| Rows | 10,000 customers |
| Features | 13 (CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary) |
| Target | `Exited` (1 = churned, 0 = stayed) |
| Class imbalance | ~20% positive (churned) |

---

## Top Data Insights

- Customers with **3–4 products** churn at a much higher rate than those with 1–2 products
- **Senior customers (Age ≥ 50)** who are inactive have the highest churn propensity
- **Zero-balance customers** show distinct churn patterns
- **Geography** (Germany vs France/Spain) significantly influences churn rate
- Active members churn at roughly half the rate of inactive members

---

## Engineered Features

Domain-informed features created from raw attributes (inspired by Géron Ch. 2 feature engineering principles):

| Feature | Formula | Business Signal |
|---|---|---|
| `is_zero_balance` | Balance == 0 | Disengagement signal |
| `balance_per_product` | Balance / NumOfProducts | Wallet share per product |
| `balance_salary_ratio` | Balance / EstimatedSalary | Relative wealth commitment |
| `age_tenure_ratio` | Age / (Tenure + 1) | Life stage vs loyalty |
| `senior_inactive` | Age ≥ 50 AND IsActiveMember == 0 | High-risk segment flag |
| `products_active_combo` | NumOfProducts × IsActiveMember | Engagement depth |
| `credit_score_band` | Binned CreditScore | Risk category |

---

## Model Performance

Training uses a **Gradient Boosting Classifier** selected via 5-fold stratified cross-validation against Logistic Regression, Random Forest, and Extra Trees.

**Cross-validation model comparison (5-fold StratifiedKFold):**

| Model | ROC AUC | Avg Precision | F1 |
|---|---|---|---|
| **Gradient Boosting** | **0.862 ± 0.006** | **0.700** | **0.596** |
| Random Forest | 0.847 ± 0.011 | 0.653 | 0.578 |
| Extra Trees | 0.837 ± 0.005 | 0.636 | 0.548 |
| Logistic Regression | 0.798 ± 0.013 | 0.550 | 0.517 |

**Final model metrics (held-out 20% test set):**

| Metric | Value |
|---|---|
| ROC AUC | 0.862 |
| Average Precision | 0.703 |
| Lift@500 | 2.75× |
| Lift@1000 | 1.79× |

> Exact values depend on the random seed. Re-run `make train` to reproduce.

---

## Quick Start for Recruiters

### Option A — Google Colab (no local setup needed)

```python
# 1. Clone and install
REPO = "https://github.com/<your-username>/client-churn-prediction.git"
!git clone {REPO} project && %cd project
!pip install -q -r requirements.txt

# 2. Download data
from google.colab import files
files.upload()          # upload kaggle.json
!mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
!mkdir -p data/raw
!kaggle datasets download -d mervetorkan/churndataset --unzip -p data/raw
!mv data/raw/Churn_Modelling.csv data/raw/churn.csv 2>/dev/null || true

# 3. Run the full pipeline
!PYTHONPATH=src python -m client_churn_prediction.cli profile
!PYTHONPATH=src python -m client_churn_prediction.cli train

# 4. Inspect results
import json, pathlib
print(json.loads(pathlib.Path("reports/metrics.json").read_text()))
```

### Option B — Local (Docker-free)

```bash
# 1. Clone
git clone https://github.com/<your-username>/client-churn-prediction.git
cd client-churn-prediction

# 2. Setup environment
make setup
source .venv/bin/activate

# 3. Download data (requires Kaggle CLI configured)
make data
# — or manually place data/raw/churn.csv

# 4. Run pipeline
make profile    # profile raw data
make train      # train model → saves models/model.joblib + reports/metrics.json
make predict    # score all customers → data/processed/predictions.csv

# 5. Start REST API
make api
# → http://localhost:8000/docs (Swagger UI)
```

### Option C — One command (all steps)

```bash
make run-all
```

---

## REST API

After `make train`, start the API with `make api` and call it:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "records": [{
      "CreditScore": 619,
      "Geography": "France",
      "Gender": "Female",
      "Age": 42,
      "Tenure": 2,
      "Balance": 0.0,
      "NumOfProducts": 1,
      "HasCrCard": 1,
      "IsActiveMember": 1,
      "EstimatedSalary": 101349.76
    }]
  }'
```

**Response:**
```json
{
  "prediction": [1],
  "score": [0.73]
}
```

Swagger docs: `http://localhost:8000/docs`

---

## Repository Structure

```
client-churn-prediction/
├── configs/
│   └── project.toml          # project contract: data, model, evaluation params
├── data/
│   ├── raw/churn.csv          # source data (not tracked in git)
│   ├── interim/               # intermediate processed files
│   └── processed/predictions.csv
├── models/
│   └── model.joblib           # serialized sklearn Pipeline
├── notebooks/
│   ├── 01_business_understanding.ipynb
│   ├── 02_data_exploration.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_comparison.ipynb
│   ├── 05_hyperparameter_tuning.ipynb
│   └── 06_deployment.ipynb
├── reports/
│   ├── metrics.json           # validation metrics
│   └── data_profile.json      # data profiling output
├── scripts/
│   └── sample_api_request.py  # API consumption example
├── src/
│   └── client_churn_prediction/
│       ├── api.py             # FastAPI application
│       ├── cli.py             # CLI entry-points
│       ├── config.py          # config loader
│       ├── data.py            # data loading & profiling
│       ├── features.py        # feature engineering (churn domain)
│       └── models.py          # training, CV comparison, tuning, prediction
├── tests/
├── Makefile                   # automation commands
├── Procfile                   # cloud deployment process file
├── requirements.txt
└── requirements-api.txt
```

---

## Running the Notebooks

```bash
make notebooks
# Opens JupyterLab at http://localhost:8888
# Run notebooks in order: 01 → 02 → 03 → 04 → 05 → 06
```

---

## Make Commands Reference

| Command | Description |
|---|---|
| `make setup` | Create venv and install all dependencies |
| `make profile` | Profile raw data → `reports/data_profile.json` |
| `make train` | Train model → `models/model.joblib` + `reports/metrics.json` |
| `make compare` | Cross-validate 4 models and print comparison table |
| `make tune` | Run RandomizedSearchCV and print best hyperparameters |
| `make predict` | Batch score all customers → `data/processed/predictions.csv` |
| `make api` | Start FastAPI server at `localhost:8000` |
| `make test` | Run test suite |
| `make run-all` | Profile + train + predict in one shot |
| `make clean` | Remove caches and compiled files |

---

## Tech Stack

| Layer | Library |
|---|---|
| Data manipulation | pandas, numpy |
| ML pipeline | scikit-learn (Pipeline, ColumnTransformer) |
| Models | GradientBoostingClassifier, RandomForest, LogisticRegression |
| Hyperparameter tuning | RandomizedSearchCV (scikit-learn) |
| Model serialization | joblib |
| REST API | FastAPI + Uvicorn |
| Visualization | matplotlib, seaborn |
| Config | TOML |
| Testing | pytest |

---

## References

- Géron, A. (2025). *Hands-On Machine Learning with Scikit-Learn and PyTorch*. O'Reilly Media.
- Dataset: [Churn Modelling — Kaggle](https://www.kaggle.com/datasets/mervetorkan/churndataset)

---

## Next Steps

- [ ] Add SHAP explainability for individual predictions
- [ ] Build gain/lift curves by decile
- [ ] Estimate campaign ROI per contact budget
- [ ] Publish batch-scoring endpoint for full customer portfolio
- [ ] Add model monitoring with data drift detection

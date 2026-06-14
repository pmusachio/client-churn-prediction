# Client Churn Prediction — Retention Ranking

> Imbalanced classification · Churn propensity ranking · Lift and cumulative gains

## Business Problem

A retail bank loses revenue when customers close their accounts. Retention offers (fee waivers,
rate improvements, outreach) cost money, so they cannot be sent to everyone. The decision the
model informs is **which customers to target for retention first**: it scores each customer's
churn probability and the team works the ranked list down to its budget.

The cost of error is asymmetric. A false positive spends a retention incentive on a customer who
would have stayed anyway; a false negative loses a customer who could have been saved, forfeiting
their lifetime value. Because the retention budget is the binding constraint, the model is judged
on **lift at capacity** rather than raw accuracy.

A static rule ("target everyone in Germany with one product") was rejected: it captures part of
the signal but cannot weigh age, activity, balance and tenure together the way the model does.

## Dataset

[Churn Modelling dataset](https://www.kaggle.com/datasets/mervetorkan/churndataset)

| Property | Value |
|----------|-------|
| Rows | 10,000 customers |
| Target | `Exited` (1 = closed the account) |
| Positive rate | 20.4% |
| Key features | `Age`, `NumOfProducts`, `Balance`, `IsActiveMember`, `Geography`, `Tenure` |

## Solution Strategy

1. **Acquisition** — pull the dataset from Kaggle on demand; a versioned stratified sample backs an offline run.
2. **Leakage control** — every attribute is observed while the customer is still active, so there is no target leakage; the `RowNumber`, `CustomerId` and `Surname` identifiers are dropped.
3. **Feature engineering and encoding** — engineered ratios plus binary `Gender` and one-hot `Geography`, all inside the model `Pipeline` so serving reuses the exact transform.
4. **Imbalance** — handled with `class_weight="balanced"` inside the fitted folds only.
5. **Model selection** — `StratifiedKFold` cross-validation compares a logistic baseline, random forest and histogram gradient boosting on ROC AUC; the winner is tuned with `RandomizedSearchCV`.
6. **Evaluation** — ROC AUC and average precision on a stratified holdout, plus lift and churners captured at fixed targeting capacities and ROC AUC by segment.

## Top Insights & Hypotheses

- **Age is the strongest churn driver** (permutation importance 0.12): older customers leave more often.
- **Product holdings are nearly as important** (0.12): single-product customers are far more likely to exit, and customers with three or four products churn at very high rates.
- **Inactive members and German customers churn more**, consistent with engagement and market effects; the model exploits these alongside balance.
- **Performance is stable across geographies** (ROC AUC 0.857-0.874), so a single model serves all three markets.

## Engineered Features

| Feature | Formula | Business signal |
|---------|---------|-----------------|
| BalanceSalaryRatio | `Balance / (EstimatedSalary + 1)` | Wealth concentration in the account relative to income. |
| ZeroBalance | `1 if Balance == 0 else 0` | A zeroed balance often precedes account closure. |
| ProductsPerTenure | `NumOfProducts / (Tenure + 1)` | Product uptake rate; rapid accumulation can signal mis-selling and churn. |

## Model

A histogram gradient boosting classifier (selected by cross-validation, tuned with randomized
search) inside a `Pipeline` that owns the engineering and encoding. The logistic baseline sets the
bar the final model must clear.

| Model | CV ROC AUC | Holdout ROC AUC | Holdout AP |
|-------|-----------:|----------------:|-----------:|
| Logistic baseline | 0.766 | 0.776 | — |
| Random forest | 0.854 | — | — |
| **Hist gradient boosting (final)** | **0.854** | **0.870** | **0.720** |

## Business Results

Ranking the holdout by churn score and targeting top-N customers:

| Targeted | Precision@k | Lift vs random | Churners captured |
|----------|------------:|---------------:|------------------:|
| 200 | 84.0% | 4.12x | 41.3% |
| 500 | 57.8% | 2.84x | 71.0% |
| 1,000 | 36.6% | 1.80x | 89.9% |

Targeting the 200 highest-risk customers reaches **41% of all churners at 4.1x** the efficiency of
random outreach; extending to the top 1,000 captures **90%** of churn. The bank can size the
campaign to its budget and read the expected coverage straight off the curve.

## How to Run

1. **Clone**
   ```
   git clone https://github.com/pmusachio/client-churn-prediction.git
   cd client-churn-prediction
   ```
2. **Environment**
   ```
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. **Kaggle access** — place a Kaggle API token at `~/.kaggle/`; the pipeline falls back to the versioned sample if none is present.
4. **Run the pipeline**
   ```
   python -m src.pipeline
   ```
5. **Tests**
   ```
   pytest tests/
   ```
6. **App (local)**
   ```
   streamlit run app/streamlit_app.py
   ```
7. **Live app** — [huggingface.co/spaces/pmusachio/client-churn-prediction](https://huggingface.co/spaces/pmusachio/client-churn-prediction) — score a customer and explore the retention campaign view.

## Next Steps

- Add behavioural and transaction-trend features (declining balance, falling activity) which tend to lead churn and are not in the current snapshot.
- Calibrate probabilities and attach a customer-lifetime-value estimate so targeting maximizes expected saved value rather than raw churn probability; deferred until CLV inputs are available.
- Monitor for drift and refit on a schedule, since churn behaviour shifts with product and pricing changes; deferred to a deployment phase.

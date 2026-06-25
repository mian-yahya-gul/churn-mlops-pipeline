# Customer Churn Prediction  MLOps Pipeline

An end to end machine learning pipeline for predicting customer churn, built to demonstrate production MLOps practices: experiment tracking, model registry, containerization, automated testing, and CI/CD  not just a model training script.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![MLflow](https://img.shields.io/badge/MLflow-2.14-0194E2)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED)
![CI](https://github.com/mian-yahya-gul/churn-mlops-pipeline/actions/workflows/ci.yml/badge.svg)

## Problem

Subscriptionmbased businesses lose revenue when customers churn. This pipeline trains a classifier to flag at risk customers using account and usage data (tenure, contract type, billing, support interactions), so retention teams can act before a customer leaves.

## What this project demonstrates

This is a full lifecycle a production ML system needs:

- **Reproducible preprocessing**  deterministic train/test splits, persisted label encoders so inference matches training exactly
- **Experiment tracking with MLflow** — every run logs parameters, metrics, and artifacts; multiple hyperparameter configurations are compared side by side in the MLflow UI
- **Model registry** — trained models are versioned and registered (`churn-rf-classifier`), so inference always pulls a specific, traceable version
- **Automated testing** — unit tests validate data integrity and preprocessing correctness, run automatically in CI
- **Containerization** — the full pipeline (and an MLflow UI) runs via Docker Compose, no local environment setup needed
- **CI/CD** — GitHub Actions runs tests, a training smoke test, and a Docker build on every push

## Architecture

```
Raw Data (CSV)
      │
      ▼
 preprocess.py  ──► encodes categoricals, splits train/test, saves encoders
      │
      ▼
   train.py     ──► trains RandomForestClassifier, logs to MLflow
      │                (params, metrics, confusion matrix, feature importance)
      ▼
MLflow Model Registry  ──► versioned model artifact
      │
      ▼
  predict.py    ──► loads latest registered model, runs inference
```

## Results

Best run (150 estimators, max_depth=10):

| Metric | Score |
|---|---|
| Accuracy | 0.77 |
| Precision | 0.76 |
| Recall | 0.79 |
| F1 Score | 0.78 |
| ROC-AUC | 0.86 |

Top predictive features: `contract_type`, `total_charges`, `monthly_charges`, `tenure_months` — consistent with churn literature (month to month customers churn far more than annual contract customers).

## Running it

### Option 1 — Docker Compose (recommended)

```bash
docker-compose up --build
```

This runs the full pipeline and starts the MLflow UI at `http://localhost:5000`.

### Option 2 — Local

```bash
pip install -r requirements.txt

python src/preprocess.py        # generates train/test splits
python src/train.py             # trains + logs to MLflow
python src/predict.py           # sample inference

mlflow ui                       # view experiment runs at localhost:5000
```

### Running tests

```bash
pytest tests/ -v
```

### Comparing experiments

```bash
python src/train.py --n_estimators 300 --max_depth 15 --min_samples_split 2
python src/train.py --n_estimators 100 --max_depth 5  --min_samples_split 10
mlflow ui   # compare all runs side-by-side
```

## Tech Stack

`Python` · `scikit-learn` · `MLflow` · `Docker` · `Docker Compose` · `GitHub Actions` · `pytest` · `pandas`

## Project Structure

```
.
├── data/
│   └── customer_churn.csv
├── src/
│   ├── preprocess.py
│   ├── train.py
│   └── predict.py
├── tests/
│   └── test_pipeline.py
├── .github/workflows/ci.yml
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```


## Author

**Mian Yahya Gul** — [LinkedIn](https://www.linkedin.com/in/mian-yahya-gul/) · [GitHub](https://github.com/mian-yahya-gul)

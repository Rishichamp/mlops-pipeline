# End-to-End MLOps Pipeline

A production-grade MLOps pipeline built from scratch with all industry-standard tools.

## Architecture

- **Data Pipeline**: Apache Airflow DAGs for ingestion, validation, feature engineering
- **Experiment Tracking**: MLflow with MinIO artifact storage
- **Model Serving**: FastAPI with A/B testing and shadow mode
- **Monitoring**: Prometheus + Grafana with drift detection
- **Orchestration**: Kubernetes with HPA auto-scaling
- **CI/CD**: Docker multi-stage builds

## Stack

| Component | Tool |
|---|---|
| Workflow | Apache Airflow 2.9 |
| Tracking | MLflow 2.17 |
| Serving | FastAPI + uvicorn |
| Storage | MinIO (S3-compatible) |
| Containers | Docker + Kubernetes |
| Monitoring | Prometheus + Grafana |
| Drift | Scipy KS Test |

## Phases Built

- Phase 1: Project structure, FastAPI, Docker, tests
- Phase 2: Airflow data pipeline, feature engineering
- Phase 3: MLflow training, 10-trial HP search, model registry
- Phase 4: Kubernetes deployment, Prometheus, Grafana
- Phase 5: Drift detection, auto-retraining
- Phase 6: A/B testing (90/10 split), shadow mode, champion-challenger

## Quick Start

```powershell
docker compose up -d
# API    → http://localhost:8000/docs
# MLflow → http://localhost:5000
# Airflow→ http://localhost:8080  (admin/admin)
# Grafana→ http://localhost:3000  (admin/admin)
```

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `MODEL_PATH` | `model.pkl` | Champion model file |
| `CHALLENGER_MODEL_PATH` | same as `MODEL_PATH` | Challenger for A/B / shadow mode |
| `AB_MODE`, `SHADOW_MODE`, `AB_SPLIT` | `false`, `false`, `0.9` | Traffic routing |
| `DRIFT_THRESHOLD` | `0.1` | Mean KS statistic that counts as drift |
| `DRIFT_MIN_SAMPLES` | `30` | Fewer production samples → no decision (KS is noisy on tiny samples) |
| `DRIFT_WINDOW` | `500` | Drift compares the most recent N predictions |

`/predict` answers `422` for a wrong feature count or non-finite values and
`503` when no model is loaded.

## Tests

```bash
pip install -r requirements.txt
python -m pytest -q tests
```
If `artifacts/model.pkl` is missing the tests train a small iris model on the fly.

## Dataset

Iris classification (150 samples, 3 classes) used as a proxy.
The pipeline architecture is dataset-agnostic.

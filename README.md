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
\\\powershell
docker compose up -d
# API    → http://localhost:8000/docs
# MLflow → http://localhost:5000
# Airflow→ http://localhost:8080  (admin/admin)
# Grafana→ http://localhost:3000  (admin/admin)
\\\

## Dataset
Iris classification (150 samples, 3 classes) used as a proxy.
The pipeline architecture is dataset-agnostic.

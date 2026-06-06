# =============================================================================
# MLOps Pipeline — Complete Reference Script
# Project: R:\Projects\mlops-pipeline
# Python: 3.13 | OS: Windows 11 | Shell: PowerShell
# =============================================================================
# HOW TO USE THIS SCRIPT:
#   Do NOT run the entire file at once.
#   Each section is a standalone block. Copy-paste the section you need.
#   Sections marked [ONE-TIME] only need to run once ever.
#   Sections marked [EVERY SESSION] must run each new terminal session.
#   Sections marked [ON RESTART] must run after every docker compose up.
# =============================================================================


# =============================================================================
# SECTION 0 — EVERY SESSION SETUP
# Run these 3 lines at the start of every new PowerShell terminal.
# =============================================================================

cd R:\Projects\mlops-pipeline
.\venv\Scripts\Activate.ps1
$env:Path += ";$env:USERPROFILE\tools"   # adds minikube + kubectl to PATH


# =============================================================================
# SECTION 1 — PROJECT STRUCTURE CREATION  [ONE-TIME]
# Creates the full folder tree and placeholder files.
# =============================================================================

mkdir mlops-pipeline
cd mlops-pipeline

$folders = @(
    "src\ingestion", "src\features", "src\training",
    "src\evaluation", "src\serving", "src\monitoring",
    "dags", "docker", "k8s", "mlflow", "tests",
    "data\raw", "data\processed", "data\features",
    "artifacts"
)
foreach ($f in $folders) { mkdir -Force $f }

$inits = @(
    "src\__init__.py", "src\ingestion\__init__.py",
    "src\features\__init__.py", "src\training\__init__.py",
    "src\evaluation\__init__.py", "src\serving\__init__.py",
    "src\monitoring\__init__.py", "tests\__init__.py"
)
foreach ($f in $inits) { New-Item -Force $f | Out-Null }

New-Item -Force "data\raw\.gitkeep"       | Out-Null
New-Item -Force "data\processed\.gitkeep" | Out-Null
New-Item -Force "data\features\.gitkeep"  | Out-Null

git init
git checkout -b main


# =============================================================================
# SECTION 2 — PYTHON VIRTUAL ENVIRONMENT  [ONE-TIME]
# Python 3.13 requires newer package versions — pinned versions below are
# the ones confirmed working. pyarrow does NOT install on Python 3.13;
# use fastparquet everywhere instead.
# =============================================================================

# Allow script execution (run once if you get execution policy errors)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

python -m venv venv
.\venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install -r requirements.txt

# MLflow must be installed separately as mlflow-skinny to avoid pyarrow dependency
pip install mlflow-skinny==2.17.2

# Parquet support — pyarrow won't build on Python 3.13, use fastparquet
pip install fastparquet

# MinIO / S3 client
pip install boto3

# Drift detection KS test
pip install scipy

# Freeze working versions for reproducibility
pip freeze | Out-File -Encoding utf8 requirements.lock


# =============================================================================
# SECTION 3 — LOCAL ENV VARS FOR MLFLOW + MINIO  [EVERY SESSION when training]
# Set these before running any training script locally.
# The docker-compose services use internal Docker DNS (mlflow:5000, minio:9000).
# From your host machine, use localhost with the mapped ports instead.
# =============================================================================

$env:MLFLOW_TRACKING_URI    = "http://localhost:5000"
$env:MLFLOW_EXPERIMENT_NAME = "mlops-pipeline"
$env:MLFLOW_S3_ENDPOINT_URL = "http://localhost:9000"
$env:AWS_ACCESS_KEY_ID      = "minioadmin"
$env:AWS_SECRET_ACCESS_KEY  = "minioadmin"
$env:AWS_DEFAULT_REGION     = "us-east-1"


# =============================================================================
# SECTION 4 — TRAIN BASELINE MODEL  [RUN WHEN model.pkl IS MISSING]
# Produces artifacts\model.pkl — required before running the API or
# building the Docker image. Re-run after deleting artifacts\.
# =============================================================================

python src\training\train_baseline.py
# Expected output:
#   INFO:__main__:Test accuracy: 1.0000
#   INFO:__main__:Saved to artifacts/model.pkl


# =============================================================================
# SECTION 5 — RUN API LOCALLY (no Docker)
# Useful for fast development. Stop with Ctrl+C before running pytest.
# =============================================================================

$env:MODEL_PATH = "artifacts\model.pkl"
uvicorn src.serving.app:app --reload --port 8000

# Test in a second terminal:
Invoke-RestMethod http://localhost:8000/health
Invoke-RestMethod -Method Post `
  -Uri http://localhost:8000/predict `
  -ContentType "application/json" `
  -Body '{"features": [5.1, 3.5, 1.4, 0.2]}'


# =============================================================================
# SECTION 6 — PYTEST
# Stop uvicorn (Ctrl+C) before running tests.
# conftest.py sets MODEL_PATH using absolute path — this is intentional.
# The module-level TestClient(app) pattern breaks because lifespan fires
# before env vars are set; the fixture pattern fixes this.
# =============================================================================

pytest tests\ -v


# =============================================================================
# SECTION 7 — DOCKER IMAGE BUILD
# Rebuild whenever src\serving\app.py or requirements.txt changes.
# The two-stage Dockerfile keeps the final image small by discarding
# the builder stage (pip, wheel files, compilers).
# Docker Desktop must be open with green "Engine running" status.
# =============================================================================

docker build -f docker\serving.Dockerfile -t mlops-serving:latest .

# Verify image was created
docker images | Select-String "mlops-serving"

# Run container locally to test
docker run --rm -p 8000:8000 mlops-serving:latest

# Test from second terminal
Invoke-RestMethod http://localhost:8000/health


# =============================================================================
# SECTION 8 — DOCKER COMPOSE (full stack)
# Starts: FastAPI(8000) + MLflow(5000) + MinIO(9001) + Airflow(8080)
#         + Postgres(5432) + Prometheus(9090) + Grafana(3000)
# airflow-init container will show "Exited (0)" — this is correct,
# it is a one-time DB migration job.
# =============================================================================

docker compose up -d

# Check all containers are healthy
docker compose ps

# View logs for a specific service
docker compose logs -f serving
docker compose logs -f airflow-scheduler

# Stop everything (volumes are preserved)
docker compose down

# Stop and DELETE all volumes (wipes MLflow DB, MinIO data, Airflow DB)
docker compose down -v   # use only when you want a full reset


# =============================================================================
# SECTION 9 — AIRFLOW PACKAGE INSTALL  [ON RESTART — after every docker compose up]
# IMPORTANT: Airflow containers lose manually-installed packages on restart.
# The permanent fix is a custom Dockerfile inheriting from apache/airflow:2.9.2
# with RUN pip install ... baked in. The workaround below is used here.
# Must run BOTH scheduler and webserver.
# =============================================================================

docker exec -u 50000 mlops-airflow-scheduler python -m pip install `
    scikit-learn pandera boto3 fastparquet mlflow-skinny==2.17.2 scipy --quiet

docker exec -u 50000 mlops-airflow-webserver python -m pip install `
    scikit-learn pandera boto3 fastparquet mlflow-skinny==2.17.2 scipy --quiet

# Verify
docker exec mlops-airflow-scheduler python -c `
    "import sklearn, mlflow, pandera, boto3, fastparquet, scipy; print('All OK')"


# =============================================================================
# SECTION 10 — MINIO BUCKET SETUP  [ONE-TIME or after docker compose down -v]
# Creates the three buckets required by MLflow, features pipeline, and
# drift detection. Run after MinIO starts.
# =============================================================================

python -c "
import sys; sys.path.insert(0, '.')
from src.ingestion.minio_client import ensure_bucket
ensure_bucket('mlflow-artifacts')   # MLflow model artifacts
ensure_bucket('mlops-features')     # Feature parquet files
ensure_bucket('mlops-models')       # Exported model files
ensure_bucket('mlops-predictions')  # Production prediction logs
print('All 4 buckets ready')
"


# =============================================================================
# SECTION 11 — RUN TRAINING LOCALLY (MLflow + MinIO must be running)
# Runs 10 hyperparameter trials and logs everything to MLflow.
# Check results at http://localhost:5000 -> Experiments -> mlops-pipeline
# =============================================================================

python src\training\train_mlflow.py

# Expected output (abridged):
#   INFO: Run xxxxxxxx  random_forest  metrics={'accuracy': 0.9667, ...}
#   INFO: Best run: xxxxxxxx  f1_macro=1.0
#   INFO: Registered version N  status=READY
#   INFO: Model iris-classifier vN promoted to Production


# =============================================================================
# SECTION 12 — DATA PIPELINE MODULES (test individually)
# Run these to verify each module works before running in Airflow.
# =============================================================================

# Step 1: Ingest raw data
python -m src.ingestion.data_loader
# Output: INFO: Saved 150 rows to data/raw/iris.csv

# Step 2: Validate schema
python -m src.ingestion.validator
# Output: INFO: Validation passed

# Step 3: Engineer features (produces data/features/features.parquet)
python -m src.features.feature_engineering
# Output: INFO: Feature engineering complete. Shape: (150, 10)

# Step 4: Drift detection (generates synthetic drifted data, runs KS test)
python -m src.monitoring.drift_detector
# Output: Drift score: 0.9226  Drift detected: True


# =============================================================================
# SECTION 13 — AIRFLOW UI — DAG OPERATIONS
# Access at http://localhost:8080  login: admin / admin
# Two DAGs exist in the project:
#   1. mlops_data_pipeline     — daily, runs full ingest→train pipeline
#   2. drift_detection_dag     — daily, computes drift and auto-retrains
#   3. champion_challenger_dag — weekly, trains challenger and compares to champion
#
# Trigger a DAG manually from PowerShell:
# =============================================================================

docker exec mlops-airflow-scheduler airflow dags trigger mlops_data_pipeline
docker exec mlops-airflow-scheduler airflow dags trigger drift_detection_dag
docker exec mlops-airflow-scheduler airflow dags trigger champion_challenger_dag

# List all DAGs
docker exec mlops-airflow-scheduler airflow dags list

# Check DAG run status
docker exec mlops-airflow-scheduler airflow dags list-runs -d mlops_data_pipeline


# =============================================================================
# SECTION 14 — PROMETHEUS + GRAFANA
# Prometheus: http://localhost:9090
#   - Status -> Targets -> mlops-serving should show State: UP
#   - Query examples:
#       rate(http_requests_total[1m])
#       histogram_quantile(0.99, rate(http_request_duration_seconds_bucket[1m]))
#       mlops_drift_score
#
# Grafana: http://localhost:3000  login: admin / admin
#   - Add datasource: Connections -> Data sources -> Prometheus
#   - URL: http://prometheus:9090  (internal Docker DNS, not localhost)
#
# Generate load to see live metrics:
# =============================================================================

$body = '{"features": [5.1, 3.5, 1.4, 0.2]}'
1..50 | ForEach-Object {
    Invoke-RestMethod -Method Post -Uri http://localhost:8000/predict `
      -ContentType "application/json" -Body $body | Out-Null
    Start-Sleep -Milliseconds 200
}
Write-Host "50 predictions sent — check Grafana at http://localhost:3000"

# Reload Prometheus config without restart (after editing monitoring\prometheus.yml)
Invoke-RestMethod -Method Post http://localhost:9090/-/reload


# =============================================================================
# SECTION 15 — KUBERNETES (minikube)
# minikube uses Docker Desktop as its driver.
# The cluster persists between reboots but must be started each session.
# =============================================================================

# Start cluster (2GB RAM — Docker Desktop has ~3.5GB allocated)
minikube start --driver=docker --cpus=2 --memory=2048

# Check cluster is healthy
minikube status
kubectl get nodes

# Load local Docker image into minikube
# Required because minikube has its own container registry separate from Docker Desktop
minikube image load mlops-serving:latest

# Apply all manifests (namespace must exist before deployment)
kubectl apply -f k8s\namespace.yaml
kubectl apply -f k8s\deployment.yaml
kubectl apply -f k8s\service.yaml
kubectl apply -f k8s\hpa.yaml

# Watch pods come up — Ctrl+C when both show 1/1 Running
kubectl get pods -n mlops -w

# Check all K8s resources
kubectl get all -n mlops


# =============================================================================
# SECTION 16 — KUBERNETES SERVICE ACCESS (Windows tunnel)
# IMPORTANT: On Windows with Docker driver, minikube service --url creates
# a TCP tunnel. The terminal running this command MUST stay open.
# Open a SECOND terminal for testing.
# =============================================================================

# Terminal 1 — keep open, note the URL printed
minikube service mlops-serving-service -n mlops --url
# Output: http://127.0.0.1:XXXXX  ← use this port in Terminal 2

# Terminal 2 — replace PORT with the number from Terminal 1
$MINIKUBE_PORT = "REPLACE_WITH_PORT"
Invoke-RestMethod "http://127.0.0.1:$MINIKUBE_PORT/health"
Invoke-RestMethod -Method Post "http://127.0.0.1:$MINIKUBE_PORT/predict" `
  -ContentType "application/json" `
  -Body '{"features": [5.1, 3.5, 1.4, 0.2]}'


# =============================================================================
# SECTION 17 — KUBERNETES ROLLING UPDATE
# After rebuilding the Docker image, update the deployment.
# maxUnavailable=0 means zero-downtime — new pod starts before old one stops.
# =============================================================================

# Rebuild image with latest code
docker build -f docker\serving.Dockerfile -t mlops-serving:latest .

# Reload into minikube
minikube image load mlops-serving:latest

# Trigger rolling update by restarting the deployment
kubectl rollout restart deployment/mlops-serving -n mlops

# Watch the rollout
kubectl rollout status deployment/mlops-serving -n mlops

# Debug a pod if something goes wrong
kubectl describe pod -l app=mlops-serving -n mlops
kubectl logs -l app=mlops-serving -n mlops --tail=50


# =============================================================================
# SECTION 18 — A/B TESTING
# AB_MODE and SHADOW_MODE are controlled by environment variables.
# Default in docker-compose.yml is both false (champion-only mode).
#
# AB_MODE=true   → 90% champion, 10% challenger (configurable via AB_SPLIT)
# SHADOW_MODE=true → challenger runs silently on all requests, not returned
#
# To enable A/B mode without restarting the full stack, update docker-compose.yml
# and run: docker compose up -d serving
# =============================================================================

# Run the A/B test runner script (sends 100 requests and reports split)
python src\serving\ab_test_runner.py

# Expected output with AB_MODE=false:
#   champion    : 100 requests (100.0%)  avg_confidence=1.0000

# Enable A/B mode by editing docker-compose.yml:
#   AB_MODE: 'true'
# Then restart serving:
docker compose up -d serving
Start-Sleep -Seconds 15

# Run again — should show ~90/10 split
python src\serving\ab_test_runner.py


# =============================================================================
# SECTION 19 — GIT CHECKPOINTS
# Commit after each phase is verified working.
# =============================================================================

git add .
git commit -m "phase 1 complete: project structure, FastAPI serving, baseline model, tests"

git add .
git commit -m "phase 2 complete: airflow DAG, data validation, feature engineering, MinIO upload"

git add .
git commit -m "phase 3 complete: MLflow training, 10 trials, model registry, quality gate, full Airflow DAG"

git add .
git commit -m "phase 4 complete: prometheus scraping, grafana dashboard, alerting rules, kubernetes"

git add .
git commit -m "phase 5: drift detection, prediction logging, auto-retraining DAG, scipy KS test"

git add .
git commit -m "phase 6 complete: A/B router, shadow mode, champion-challenger DAG, auto-promotion"

# View commit history
git log --oneline


# =============================================================================
# SECTION 20 — TROUBLESHOOTING REFERENCE
# =============================================================================

# --- Docker engine not found ---
# Open Docker Desktop and wait for green "Engine running" bottom-left.
# Then retry the failing command.

# --- Airflow packages missing after restart ---
# Always run Section 9 after every docker compose up.

# --- MLflow version mismatch (404 errors in Airflow train task) ---
# Symptom: "couldn't get current server API group list" or 404 on /api/2.0/mlflow/logged-models
# Fix: pin mlflow-skinny==2.17.2 in Airflow containers (matches server v2.13.2)
docker exec -u 50000 mlops-airflow-scheduler python -m pip install mlflow-skinny==2.17.2 --force-reinstall --quiet

# --- pyarrow install fails ---
# pyarrow does not have a Python 3.13 wheel. Use fastparquet everywhere.
pip install fastparquet

# --- pytest test_predict_valid fails with 503 Model not loaded ---
# conftest.py must set MODEL_PATH before any import of src.serving.app
# The fix is already in place — if it regresses, check conftest.py sets
# os.environ['MODEL_PATH'] using os.path.abspath(__file__) based path.

# --- Pandera Check.greater_than() fails ---
# Newer Pandera versions broke the class method shortcuts on Python 3.13.
# Use element-wise lambdas instead:
#   WRONG: Check.greater_than(0)
#   RIGHT: Check(lambda x: x > 0, element_wise=True)

# --- MinIO NoCredentialsError in Airflow train task ---
# Must set these env vars at the TOP of task_train() before any mlflow import:
#   os.environ['AWS_ACCESS_KEY_ID']      = 'minioadmin'
#   os.environ['AWS_SECRET_ACCESS_KEY']  = 'minioadmin'
#   os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://minio:9000'

# --- minikube tunnel closes ---
# The terminal running `minikube service ... --url` must stay open on Windows.
# Open a new terminal for kubectl and API testing.

# --- Docker Desktop memory error for minikube ---
# minikube start --memory=4096 fails if Docker Desktop has < 4096MB.
# Fix: use --memory=2048 OR increase Docker Desktop memory:
#   Docker Desktop -> Settings -> Resources -> Memory -> 6GB

# --- version attribute warning in docker-compose.yml ---
# "the attribute version is obsolete" — harmless warning.
# Remove the top-level "version: '3.9'" line from docker-compose.yml to silence it.


# =============================================================================
# SECTION 21 — SERVICE URLS QUICK REFERENCE
# =============================================================================

# FastAPI serving API:   http://localhost:8000/docs
# FastAPI health:        http://localhost:8000/health
# FastAPI metrics:       http://localhost:8000/metrics
# MLflow UI:             http://localhost:5000
# MinIO console:         http://localhost:9001  (minioadmin / minioadmin)
# Airflow UI:            http://localhost:8080  (admin / admin)
# Prometheus:            http://localhost:9090
# Grafana:               http://localhost:3000  (admin / admin)
#
# Grafana Prometheus datasource URL (internal Docker DNS):
#   http://prometheus:9090   ← NOT localhost:9090


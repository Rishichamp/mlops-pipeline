# =============================================================================
# MLOps Pipeline - Startup & Reference Script
# Project: R:\Projects\mlops-pipeline
# Built through Phase 4 (FastAPI + Docker + Airflow + MLflow + MinIO + K8s + Grafana)
# =============================================================================
# HOW TO USE THIS SCRIPT:
#   Option A - Run everything at once:
#       Right-click this file -> "Run with PowerShell"
#       OR in VS Code terminal: .\mlops_startup.ps1
#
#   Option B - Run individual sections manually:
#       Copy-paste individual blocks into your VS Code terminal as needed.
#
# WHAT THIS SCRIPT DOES (in order):
#   1. Navigates to the project folder
#   2. Activates the Python virtual environment
#   3. Adds minikube/kubectl to PATH
#   4. Starts Docker Desktop (if not running)
#   5. Starts the full Docker Compose stack (8 services)
#   6. Reinstalls Airflow container packages (lost on every restart)
#   7. Starts minikube and deploys K8s pods
#   8. Runs a quick health check on all services
#   9. Prints a summary of all service URLs
# =============================================================================

# --------------------------------------------------------------------------- #
# CONFIGURATION — edit these if anything changes
# --------------------------------------------------------------------------- #
$PROJECT_DIR   = "R:\Projects\mlops-pipeline"
$TOOLS_DIR     = "$env:USERPROFILE\tools"          # where minikube.exe & kubectl.exe live
$VENV_ACTIVATE = "$PROJECT_DIR\venv\Scripts\Activate.ps1"

# Airflow packages — reinstall these every time Docker restarts (Airflow containers
# don't persist pip installs between restarts — this is a known limitation of using
# the base apache/airflow image without a custom Dockerfile)
$AIRFLOW_PACKAGES = "scikit-learn pandera boto3 fastparquet mlflow-skinny==2.17.2"

# --------------------------------------------------------------------------- #
# HELPER: print a colored section header
# --------------------------------------------------------------------------- #
function Write-Section($title) {
    Write-Host "`n$('=' * 60)" -ForegroundColor Cyan
    Write-Host "  $title" -ForegroundColor Cyan
    Write-Host "$('=' * 60)" -ForegroundColor Cyan
}

function Write-OK($msg)   { Write-Host "  [OK] $msg" -ForegroundColor Green }
function Write-WARN($msg) { Write-Host "  [!!] $msg" -ForegroundColor Yellow }
function Write-STEP($msg) { Write-Host "  --> $msg" -ForegroundColor White }

# --------------------------------------------------------------------------- #
# STEP 1: Navigate to project folder
# --------------------------------------------------------------------------- #
Write-Section "STEP 1: Navigate to project"
Set-Location $PROJECT_DIR
Write-OK "Working directory: $(Get-Location)"

# --------------------------------------------------------------------------- #
# STEP 2: Activate virtual environment
# --------------------------------------------------------------------------- #
Write-Section "STEP 2: Activate virtual environment"
if (Test-Path $VENV_ACTIVATE) {
    & $VENV_ACTIVATE
    Write-OK "venv activated (Python $(python --version))"
} else {
    Write-WARN "venv not found at $VENV_ACTIVATE"
    Write-WARN "Create it with:  python -m venv venv"
    Write-WARN "Then install:    pip install -r requirements.txt"
    Write-WARN "Then install:    pip install mlflow-skinny==2.17.2 boto3 fastparquet"
    exit 1
}

# --------------------------------------------------------------------------- #
# STEP 3: Add minikube/kubectl to PATH
# --------------------------------------------------------------------------- #
Write-Section "STEP 3: Add tools to PATH"
if (Test-Path "$TOOLS_DIR\minikube.exe") {
    $env:Path += ";$TOOLS_DIR"
    Write-OK "minikube: $(minikube version --short)"
    Write-OK "kubectl:  $(kubectl version --client --short 2>$null)"
} else {
    Write-WARN "minikube not found at $TOOLS_DIR\minikube.exe"
    Write-WARN "Download from: https://storage.googleapis.com/minikube/releases/latest/minikube-windows-amd64.exe"
    Write-WARN "Save to: $TOOLS_DIR\minikube.exe"
}

# --------------------------------------------------------------------------- #
# STEP 4: Check Docker Desktop is running
# --------------------------------------------------------------------------- #
Write-Section "STEP 4: Check Docker Desktop"
$dockerRunning = $false
try {
    $dockerInfo = docker info 2>&1
    if ($LASTEXITCODE -eq 0) {
        $dockerRunning = $true
        Write-OK "Docker Desktop is running"
    }
} catch {}

if (-not $dockerRunning) {
    Write-STEP "Starting Docker Desktop..."
    Start-Process "C:\Program Files\Docker\Docker\Docker Desktop.exe"
    Write-WARN "Waiting 60 seconds for Docker engine to start..."
    Start-Sleep -Seconds 60
    Write-OK "Docker Desktop started — if it still fails, open it manually and rerun this script"
}

# --------------------------------------------------------------------------- #
# STEP 5: Start Docker Compose stack
# --------------------------------------------------------------------------- #
Write-Section "STEP 5: Start Docker Compose stack (8 services)"
Write-STEP "Running: docker compose up -d"
docker compose up -d

Write-STEP "Waiting 30 seconds for services to initialize..."
Start-Sleep -Seconds 30

Write-STEP "Container status:"
docker compose ps

# --------------------------------------------------------------------------- #
# STEP 6: Reinstall packages in Airflow containers
# --------------------------------------------------------------------------- #
# WHY THIS IS NEEDED:
#   The project uses the stock apache/airflow:2.9.2 image which does not have
#   scikit-learn, pandera, boto3 etc. These are manually installed at runtime.
#   Docker volumes persist DATA (postgres DB, MinIO files) but NOT pip installs.
#   Every time the containers restart, pip packages are gone from the container
#   filesystem. Fix in Phase 5/6 would be a custom Airflow Dockerfile — for now
#   we reinstall on every startup.
# --------------------------------------------------------------------------- #
Write-Section "STEP 6: Reinstall packages in Airflow containers"
Write-STEP "Installing into airflow-scheduler..."
docker exec -u 50000 mlops-airflow-scheduler python -m pip install $AIRFLOW_PACKAGES --quiet
Write-OK "Scheduler packages installed"

Write-STEP "Installing into airflow-webserver..."
docker exec -u 50000 mlops-airflow-webserver python -m pip install $AIRFLOW_PACKAGES --quiet
Write-OK "Webserver packages installed"

# Verify
$checkResult = docker exec mlops-airflow-scheduler python -c "import sklearn, mlflow, pandera, boto3; print('ALL_OK')" 2>&1
if ($checkResult -like "*ALL_OK*") {
    Write-OK "Airflow package verification passed"
} else {
    Write-WARN "Airflow package check failed: $checkResult"
    Write-WARN "Try running manually: docker exec -u 50000 mlops-airflow-scheduler python -m pip install $AIRFLOW_PACKAGES"
}

# --------------------------------------------------------------------------- #
# STEP 7: Start minikube and deploy Kubernetes pods
# --------------------------------------------------------------------------- #
Write-Section "STEP 7: Start minikube + deploy K8s pods"

$minikubeStatus = minikube status 2>&1
if ($minikubeStatus -like "*Running*") {
    Write-OK "minikube already running"
} else {
    Write-STEP "Starting minikube (driver=docker, 2 CPUs, 2048MB)..."
    minikube start --driver=docker --cpus=2 --memory=2048
}

Write-STEP "Loading Docker image into minikube..."
minikube image load mlops-serving:latest
Write-OK "Image loaded"

Write-STEP "Applying K8s manifests..."
kubectl apply -f k8s\namespace.yaml  2>$null
kubectl apply -f k8s\deployment.yaml 2>$null
kubectl apply -f k8s\service.yaml    2>$null
kubectl apply -f k8s\hpa.yaml        2>$null

Write-STEP "Waiting for pods to reach Running state (up to 60s)..."
$timeout = 60
$elapsed = 0
do {
    Start-Sleep -Seconds 5
    $elapsed += 5
    $pods = kubectl get pods -n mlops 2>&1
    $ready = ($pods | Select-String "1/1\s+Running").Count
    Write-Host "    ($elapsed s) Pods ready: $ready/2" -ForegroundColor Gray
} while ($ready -lt 2 -and $elapsed -lt $timeout)

if ($ready -ge 2) {
    Write-OK "Both K8s pods are Running"
} else {
    Write-WARN "Pods not ready after ${timeout}s — check with: kubectl get pods -n mlops"
}

# --------------------------------------------------------------------------- #
# STEP 8: Health checks on all services
# --------------------------------------------------------------------------- #
Write-Section "STEP 8: Health checks"

function Test-Endpoint($name, $url) {
    try {
        $response = Invoke-RestMethod -Uri $url -TimeoutSec 5 -ErrorAction Stop
        Write-OK "$name is UP  ($url)"
        return $true
    } catch {
        Write-WARN "$name is DOWN or not ready  ($url)"
        return $false
    }
}

Test-Endpoint "FastAPI serving"  "http://localhost:8000/health"
Test-Endpoint "MLflow UI"        "http://localhost:5000"
Test-Endpoint "MinIO API"        "http://localhost:9000/minio/health/live"
Test-Endpoint "Airflow UI"       "http://localhost:8080/health"
Test-Endpoint "Prometheus"       "http://localhost:9090/-/healthy"
Test-Endpoint "Grafana"          "http://localhost:3000/api/health"

# Test a prediction
Write-STEP "Testing prediction endpoint..."
try {
    $pred = Invoke-RestMethod -Method Post `
        -Uri "http://localhost:8000/predict" `
        -ContentType "application/json" `
        -Body '{"features": [5.1, 3.5, 1.4, 0.2]}' `
        -TimeoutSec 5
    Write-OK "Prediction OK  -> $($pred.prediction_label) (confidence: $($pred.confidence))"
} catch {
    Write-WARN "Prediction failed: $_"
}

# --------------------------------------------------------------------------- #
# STEP 9: Print summary of all URLs and quick-reference commands
# --------------------------------------------------------------------------- #
Write-Section "ALL SERVICES RUNNING — URL REFERENCE"

Write-Host @"

  SERVICE              URL                           CREDENTIALS
  ─────────────────────────────────────────────────────────────────────
  FastAPI (serving)    http://localhost:8000/docs     (none)
  FastAPI (health)     http://localhost:8000/health   (none)
  FastAPI (metrics)    http://localhost:8000/metrics  (none)
  MLflow UI            http://localhost:5000          (none)
  MinIO UI             http://localhost:9001          minioadmin / minioadmin
  Airflow UI           http://localhost:8080          admin / admin
  Prometheus           http://localhost:9090          (none)
  Grafana              http://localhost:3000          admin / admin

"@ -ForegroundColor White

Write-Host "  QUICK COMMANDS (run in VS Code terminal):" -ForegroundColor Cyan
Write-Host @"

  # Make a prediction
  Invoke-RestMethod -Method Post -Uri http://localhost:8000/predict ``
    -ContentType 'application/json' -Body '{\"features\": [5.1, 3.5, 1.4, 0.2]}'

  # Watch Kubernetes pods
  kubectl get pods -n mlops -w

  # View serving container logs
  docker logs mlops-serving -f

  # View Airflow scheduler logs
  docker logs mlops-airflow-scheduler -f

  # Trigger the Airflow DAG manually (or use the UI)
  docker exec -u 50000 mlops-airflow-scheduler ``
    airflow dags trigger mlops_data_pipeline

  # Run MLflow training locally (sets all required env vars first)
  `$env:MLFLOW_TRACKING_URI    = 'http://localhost:5000'
  `$env:MLFLOW_EXPERIMENT_NAME = 'mlops-pipeline'
  `$env:MLFLOW_S3_ENDPOINT_URL = 'http://localhost:9000'
  `$env:AWS_ACCESS_KEY_ID      = 'minioadmin'
  `$env:AWS_SECRET_ACCESS_KEY  = 'minioadmin'
  `$env:AWS_DEFAULT_REGION     = 'us-east-1'
  python src\training\train_mlflow.py

  # Run pytest
  pytest tests\ -v

  # Stop everything (Docker stack only — minikube stays running)
  docker compose down

  # Stop minikube too
  minikube stop

  # Rebuild Docker image after code changes
  docker build -f docker\serving.Dockerfile -t mlops-serving:latest .

  # After rebuild, reload into minikube and rolling-update K8s
  minikube image load mlops-serving:latest
  kubectl rollout restart deployment/mlops-serving -n mlops
  kubectl rollout status deployment/mlops-serving -n mlops

"@ -ForegroundColor Gray

Write-Host "  TROUBLESHOOTING:" -ForegroundColor Cyan
Write-Host @"

  Problem: 'docker compose up' fails with 'cannot connect to daemon'
  Fix:     Open Docker Desktop manually, wait for green 'Engine running', rerun script.

  Problem: Airflow tasks fail with 'No module named sklearn'
  Fix:     Rerun Step 6 manually (packages lost on container restart — this is expected):
           docker exec -u 50000 mlops-airflow-scheduler python -m pip install scikit-learn pandera boto3 fastparquet mlflow-skinny==2.17.2
           docker exec -u 50000 mlops-airflow-webserver  python -m pip install scikit-learn pandera boto3 fastparquet mlflow-skinny==2.17.2

  Problem: Airflow train task fails with '404 Not Found' on MLflow API
  Fix:     mlflow-skinny must be ==2.17.2 to match the MLflow server v2.13.2.
           Installing without a version pin installs v3.x which breaks the API.
           Run: docker exec -u 50000 mlops-airflow-scheduler python -m pip install mlflow-skinny==2.17.2 --force-reinstall

  Problem: minikube pods stuck in 'ContainerCreating'
  Fix:     The image needs to be loaded into minikube after every Docker restart:
           minikube image load mlops-serving:latest
           kubectl rollout restart deployment/mlops-serving -n mlops

  Problem: K8s service URL only works while terminal is open
  Fix:     This is expected with 'minikube service --url' on Windows Docker driver.
           Keep that terminal open, or use port-forward instead:
           kubectl port-forward service/mlops-serving-service 8001:80 -n mlops
           Then test on http://localhost:8001/health

  Problem: pip install fails with 'pyarrow' not found
  Fix:     Use fastparquet instead (pyarrow has no Python 3.13 wheel):
           pip install fastparquet
           Never use to_parquet() without specifying engine='fastparquet' or
           ensure fastparquet is installed so pandas auto-detects it.

"@ -ForegroundColor Gray

Write-Host "$('=' * 60)" -ForegroundColor Cyan
Write-Host "  NEXT: Phase 5 — Drift detection (Evidently AI) + auto-retraining" -ForegroundColor Yellow
Write-Host "$('=' * 60)`n" -ForegroundColor Cyan

FROM python:3.11-slim AS builder
WORKDIR /build
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt && \
    pip install --no-cache-dir --prefix=/install mlflow-skinny==2.17.2 boto3 fastparquet

FROM python:3.11-slim AS runtime
RUN useradd --create-home appuser
WORKDIR /home/appuser/app
COPY --from=builder /install /usr/local
COPY src/ ./src/
COPY artifacts/model.pkl ./model.pkl
RUN chown -R appuser:appuser /home/appuser
USER appuser
ENV MODEL_PATH=model.pkl \
    MODEL_NAME=iris-classifier \
    MODEL_VERSION=1.0.0 \
    USE_MLFLOW_REGISTRY=false \
    MLFLOW_TRACKING_URI=http://mlflow:5000 \
    MLFLOW_S3_ENDPOINT_URL=http://minio:9000 \
    AWS_ACCESS_KEY_ID=minioadmin \
    AWS_SECRET_ACCESS_KEY=minioadmin \
    AWS_DEFAULT_REGION=us-east-1 \
    API_PORT=8000 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]

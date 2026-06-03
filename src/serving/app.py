from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict
from contextlib import asynccontextmanager
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Gauge
import numpy as np
import mlflow.sklearn
import mlflow
import joblib
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom Prometheus gauge for drift score
DRIFT_SCORE = Gauge('mlops_drift_score', 'Current model drift score')
PREDICTION_COUNT = Gauge('mlops_prediction_count_total', 'Total predictions served')

class PredictRequest(BaseModel):
    features: list[float]

class PredictResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    prediction: int
    prediction_label: str
    confidence: float
    model_version: str
    model_source: str

class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    status: str
    model_loaded: bool
    model_source: str
    model_version: str

class ModelStore:
    def __init__(self):
        self.model        = None
        self.version      = 'unknown'
        self.source       = 'none'
        self.label_map    = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
        self._pred_count  = 0

    def load_from_mlflow(self, model_name: str, stage: str = 'Production'):
        mlflow_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
        mlflow.set_tracking_uri(mlflow_uri)
        model_uri  = f'models:/{model_name}/{stage}'
        logger.info(f'Loading model from MLflow: {model_uri}')
        self.model   = mlflow.sklearn.load_model(model_uri)
        self.version = os.getenv('MODEL_VERSION', 'mlflow-production')
        self.source  = f'mlflow:{model_uri}'

    def load_from_file(self, path: str):
        self.model   = joblib.load(path)
        self.version = os.getenv('MODEL_VERSION', '1.0.0')
        self.source  = f'file:{path}'
        logger.info(f'Model loaded from file: {path}')

    def predict(self, features: list[float]) -> dict:
        if self.model is None:
            raise RuntimeError('Model not loaded')
        X      = np.array(features).reshape(1, -1)
        pred   = int(self.model.predict(X)[0])
        proba  = float(self.model.predict_proba(X).max())

        # Log prediction for drift detection
        try:
            from src.monitoring.prediction_logger import log_prediction
            log_prediction(features, pred, proba)
        except Exception as e:
            logger.warning(f'Failed to log prediction: {e}')

        self._pred_count += 1
        PREDICTION_COUNT.set(self._pred_count)

        return {
            'prediction':       pred,
            'prediction_label': self.label_map.get(pred, 'unknown'),
            'confidence':       round(proba, 4),
            'model_version':    self.version,
            'model_source':     self.source,
        }

model_store = ModelStore()

@asynccontextmanager
async def lifespan(app: FastAPI):
    use_mlflow = os.getenv('USE_MLFLOW_REGISTRY', 'false').lower() == 'true'
    if use_mlflow:
        try:
            model_store.load_from_mlflow(os.getenv('MODEL_NAME', 'iris-classifier'))
        except Exception as e:
            logger.warning(f'MLflow load failed: {e}. Falling back to file.')
            path = os.getenv('MODEL_PATH', 'model.pkl')
            if os.path.exists(path):
                model_store.load_from_file(path)
    else:
        path = os.getenv('MODEL_PATH', 'model.pkl')
        if os.path.exists(path):
            model_store.load_from_file(path)
    yield

app = FastAPI(title='MLOps Serving API', version='0.3.0', lifespan=lifespan)
Instrumentator().instrument(app).expose(app)

@app.get('/health', response_model=HealthResponse)
def health():
    return {
        'status':        'ok',
        'model_loaded':  model_store.model is not None,
        'model_source':  model_store.source,
        'model_version': model_store.version,
    }

@app.post('/predict', response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        return model_store.predict(req.features)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

@app.get('/')
def root():
    return {'message': 'MLOps Serving API v0.3.0', 'docs': '/docs', 'metrics': '/metrics'}

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict
from contextlib import asynccontextmanager
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Gauge, Counter
import numpy as np
import mlflow.sklearn
import mlflow
import joblib
import os
import logging
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
DRIFT_SCORE      = Gauge('mlops_drift_score',            'Current model drift score')
PREDICTION_COUNT = Gauge('mlops_prediction_count_total', 'Total predictions served')
AB_COUNTER       = Counter('mlops_ab_requests_total',    'A/B requests', ['version'])
AB_CONFIDENCE    = Gauge('mlops_ab_confidence',          'Mean confidence per version', ['version'])

class PredictRequest(BaseModel):
    features: list[float]

class PredictResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    prediction: int
    prediction_label: str
    confidence: float
    model_version: str
    model_source: str
    served_by: str

class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    status: str
    model_loaded: bool
    model_source: str
    model_version: str
    ab_mode: bool
    shadow_mode: bool

class SingleModel:
    def __init__(self, name: str):
        self.name      = name
        self.model     = None
        self.version   = 'unknown'
        self.source    = 'none'
        self.label_map = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}

    def load_from_file(self, path: str):
        self.model   = joblib.load(path)
        self.version = os.getenv('MODEL_VERSION', '1.0.0')
        self.source  = f'file:{path}'
        logger.info(f'[{self.name}] loaded from {path}')

    def load_from_mlflow(self, model_name: str, stage: str):
        mlflow_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
        mlflow.set_tracking_uri(mlflow_uri)
        uri        = f'models:/{model_name}/{stage}'
        self.model = mlflow.sklearn.load_model(uri)
        self.version = stage
        self.source  = f'mlflow:{uri}'
        logger.info(f'[{self.name}] loaded from MLflow {uri}')

    def predict(self, features: list[float]) -> dict:
        X      = np.array(features).reshape(1, -1)
        pred   = int(self.model.predict(X)[0])
        proba  = float(self.model.predict_proba(X).max())
        return {
            'prediction':       pred,
            'prediction_label': self.label_map.get(pred, 'unknown'),
            'confidence':       round(proba, 4),
            'model_version':    self.version,
            'model_source':     self.source,
        }

class ABRouter:
    def __init__(self):
        self.champion   = SingleModel('champion')
        self.challenger = SingleModel('challenger')
        self.ab_mode     = False
        self.shadow_mode = False
        self.split       = float(os.getenv('AB_SPLIT', '0.9'))  # 90% champion
        self._pred_count = 0

    def setup(self):
        model_path = os.getenv('MODEL_PATH', 'model.pkl')
        if os.path.exists(model_path):
            self.champion.load_from_file(model_path)
            self.challenger.load_from_file(model_path)  # same model for demo

        self.ab_mode     = os.getenv('AB_MODE',     'false').lower() == 'true'
        self.shadow_mode = os.getenv('SHADOW_MODE', 'false').lower() == 'true'
        logger.info(f'AB_MODE={self.ab_mode}  SHADOW_MODE={self.shadow_mode}  SPLIT={self.split}')

    def route(self, features: list[float]) -> dict:
        self._pred_count += 1
        PREDICTION_COUNT.set(self._pred_count)

        # Shadow mode: run challenger silently alongside champion
        if self.shadow_mode and self.challenger.model:
            try:
                shadow_result = self.challenger.predict(features)
                logger.debug(f'Shadow prediction: {shadow_result["prediction"]}')
                AB_COUNTER.labels(version='shadow').inc()
            except Exception as e:
                logger.warning(f'Shadow prediction failed: {e}')

        # A/B mode: split traffic
        if self.ab_mode and self.challenger.model:
            if random.random() > self.split:
                result = self.challenger.predict(features)
                result['served_by'] = 'challenger'
                AB_COUNTER.labels(version='challenger').inc()
                AB_CONFIDENCE.labels(version='challenger').set(result['confidence'])
            else:
                result = self.champion.predict(features)
                result['served_by'] = 'champion'
                AB_COUNTER.labels(version='champion').inc()
                AB_CONFIDENCE.labels(version='champion').set(result['confidence'])
        else:
            result = self.champion.predict(features)
            result['served_by'] = 'champion'
            AB_COUNTER.labels(version='champion').inc()

        # Log prediction for drift detection
        try:
            from src.monitoring.prediction_logger import log_prediction
            log_prediction(features, result['prediction'], result['confidence'])
        except Exception as e:
            logger.warning(f'Prediction logging failed: {e}')

        return result

router = ABRouter()

@asynccontextmanager
async def lifespan(app: FastAPI):
    router.setup()
    yield

app = FastAPI(title='MLOps Serving API', version='0.4.0', lifespan=lifespan)
Instrumentator().instrument(app).expose(app)

@app.get('/health', response_model=HealthResponse)
def health():
    return {
        'status':        'ok',
        'model_loaded':  router.champion.model is not None,
        'model_source':  router.champion.source,
        'model_version': router.champion.version,
        'ab_mode':       router.ab_mode,
        'shadow_mode':   router.shadow_mode,
    }

@app.post('/predict', response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        return router.route(req.features)
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

@app.get('/')
def root():
    return {
        'message':     'MLOps Serving API v0.4.0',
        'docs':        '/docs',
        'metrics':     '/metrics',
        'ab_mode':     router.ab_mode,
        'shadow_mode': router.shadow_mode,
    }

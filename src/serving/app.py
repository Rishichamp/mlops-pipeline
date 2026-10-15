from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator
from contextlib import asynccontextmanager
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Gauge, Counter
import numpy as np
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
    features: list[float] = Field(..., min_length=1, max_length=100)

    @field_validator('features')
    @classmethod
    def finite(cls, v):
        if not all(np.isfinite(x) for x in v):
            raise ValueError('features must be finite numbers')
        return v


class ModelNotLoaded(Exception):
    pass

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
        # Imported here: only registry loading needs MLflow, so file-based
        # serving and the tests run without it installed.
        import mlflow
        import mlflow.sklearn
        mlflow_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
        mlflow.set_tracking_uri(mlflow_uri)
        uri        = f'models:/{model_name}/{stage}'
        self.model = mlflow.sklearn.load_model(uri)
        self.version = stage
        self.source  = f'mlflow:{uri}'
        logger.info(f'[{self.name}] loaded from MLflow {uri}')

    def predict(self, features: list[float]) -> dict:
        if self.model is None:
            raise ModelNotLoaded(f'No {self.name} model is loaded')
        expected = getattr(self.model, 'n_features_in_', None)
        if expected is not None and len(features) != expected:
            raise ValueError(f'Expected {expected} features, got {len(features)}')
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
        # The challenger defaults to the champion's file (demo); point
        # CHALLENGER_MODEL_PATH at a second model for a real A/B test.
        challenger_path = os.getenv('CHALLENGER_MODEL_PATH', model_path)
        if os.path.exists(model_path):
            self.champion.load_from_file(model_path)
        else:
            logger.warning(f'Champion model not found at {model_path}')
        if os.path.exists(challenger_path):
            self.challenger.load_from_file(challenger_path)

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


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError):
    # The default handler echoes the rejected input, and a NaN/Infinity input
    # can't be serialised to JSON, so the 422 itself crashed into a 500.
    errors = [{'loc': e.get('loc'), 'msg': e.get('msg'), 'type': e.get('type')}
              for e in exc.errors()]
    return JSONResponse(status_code=422, content={'detail': errors})
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
    # Previously every error became a 503 carrying the raw exception text
    # (e.g. "'NoneType' object has no attribute 'predict'").
    try:
        return router.route(req.features)
    except ModelNotLoaded as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception:
        logger.exception('Prediction failed')
        raise HTTPException(status_code=500, detail='Prediction failed')

@app.get('/')
def root():
    return {
        'message':     'MLOps Serving API v0.4.0',
        'docs':        '/docs',
        'metrics':     '/metrics',
        'ab_mode':     router.ab_mode,
        'shadow_mode': router.shadow_mode,
    }

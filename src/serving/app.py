from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict
from contextlib import asynccontextmanager
import numpy as np
import joblib
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PredictRequest(BaseModel):
    features: list[float]

class PredictResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    prediction: int
    prediction_label: str
    confidence: float
    model_version: str

class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    status: str
    model_loaded: bool

class ModelStore:
    def __init__(self):
        self.model = None
        self.version = 'unknown'
        self.label_map = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}

    def load(self, path: str):
        self.model = joblib.load(path)
        self.version = os.getenv('MODEL_VERSION', '1.0.0')
        logger.info(f'Model loaded from {path}, version={self.version}')

    def predict(self, features):
        if self.model is None:
            raise RuntimeError('Model not loaded')
        X = np.array(features).reshape(1, -1)
        pred = int(self.model.predict(X)[0])
        proba = float(self.model.predict_proba(X).max())
        return {
            'prediction': pred,
            'prediction_label': self.label_map[pred],
            'confidence': round(proba, 4),
            'model_version': self.version,
        }

model_store = ModelStore()

@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = os.getenv('MODEL_PATH', 'model.pkl')
    if os.path.exists(model_path):
        model_store.load(model_path)
    else:
        logger.warning(f'No model at {model_path}')
    yield

app = FastAPI(title='MLOps Serving API', version='0.1.0', lifespan=lifespan)

@app.get('/health', response_model=HealthResponse)
def health():
    return {'status': 'ok', 'model_loaded': model_store.model is not None}

@app.post('/predict', response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        return model_store.predict(req.features)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

@app.get('/')
def root():
    return {'message': 'MLOps Serving API', 'docs': '/docs'}
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from fastapi.testclient import TestClient

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MODEL_PATH = os.path.join(ROOT, 'artifacts', 'model.pkl')

@pytest.fixture()
def client():
    # Import fresh inside fixture so MODEL_PATH env var is set first
    os.environ['MODEL_PATH'] = MODEL_PATH
    from src.serving import app as app_module
    # Force reload of model_store for each test
    app_module.model_store.model = None
    app_module.model_store.load(MODEL_PATH)
    with TestClient(app_module.app) as c:
        yield c

def test_health(client):
    r = client.get('/health')
    assert r.status_code == 200
    assert r.json()['status'] == 'ok'

def test_predict_valid(client):
    r = client.post('/predict', json={'features': [5.1, 3.5, 1.4, 0.2]})
    assert r.status_code == 200, f'Got: {r.json()}'
    data = r.json()
    assert data['prediction'] in [0, 1, 2]
    assert 0.0 <= data['confidence'] <= 1.0
    assert data['prediction_label'] in ['setosa', 'versicolor', 'virginica']

def test_predict_wrong_input(client):
    r = client.post('/predict', json={'features': 'not_a_list'})
    assert r.status_code == 422

def test_root(client):
    r = client.get('/')
    assert r.status_code == 200

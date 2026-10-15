import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from fastapi.testclient import TestClient

from tests.conftest import MODEL_PATH

@pytest.fixture()
def client():
    # Import fresh inside fixture so MODEL_PATH env var is set first
    os.environ['MODEL_PATH'] = MODEL_PATH
    from src.serving import app as app_module
    # The app's lifespan (router.setup) loads the model from MODEL_PATH.
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


def test_predict_wrong_feature_count_is_422(client):
    r = client.post('/predict', json={'features': [5.1, 3.5]})
    assert r.status_code == 422
    assert 'Expected 4 features' in r.json()['detail']


def test_predict_rejects_non_finite(client):
    r = client.post('/predict', content='{"features": [NaN, 1, 2, 3]}',
                    headers={'Content-Type': 'application/json'})
    assert r.status_code == 422


def test_no_model_is_503_without_internals(monkeypatch):
    monkeypatch.setenv('MODEL_PATH', '/nonexistent/model.pkl')
    monkeypatch.setenv('CHALLENGER_MODEL_PATH', '/nonexistent/model.pkl')
    from src.serving import app as app_module
    app_module.router.champion.model = None
    with TestClient(app_module.app) as c:
        r = c.post('/predict', json={'features': [5.1, 3.5, 1.4, 0.2]})
    assert r.status_code == 503
    assert 'NoneType' not in r.json()['detail']

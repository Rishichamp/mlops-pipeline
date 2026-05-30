import os
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MODEL_PATH = os.path.join(ROOT, 'artifacts', 'model.pkl')

@pytest.fixture(autouse=True)
def set_model_env(monkeypatch):
    monkeypatch.setenv('MODEL_PATH', MODEL_PATH)

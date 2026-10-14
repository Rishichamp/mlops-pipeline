import os
import tempfile
import pytest

# Keep test predictions out of the tracked data/predictions log.
os.environ.setdefault('PREDICTION_LOG_PATH',
                      os.path.join(tempfile.mkdtemp(), 'predictions.jsonl'))

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MODEL_PATH = os.path.join(ROOT, 'artifacts', 'model.pkl')

@pytest.fixture(autouse=True)
def set_model_env(monkeypatch):
    monkeypatch.setenv('MODEL_PATH', MODEL_PATH)

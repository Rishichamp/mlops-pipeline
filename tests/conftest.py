import os
import tempfile

import joblib
import pytest

# Keep test predictions out of the tracked data/predictions log.
os.environ.setdefault('PREDICTION_LOG_PATH',
                      os.path.join(tempfile.mkdtemp(), 'predictions.jsonl'))

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LOCAL_MODEL = os.path.join(ROOT, 'artifacts', 'model.pkl')


def _model_path() -> str:
    """artifacts/model.pkl is gitignored; on a fresh clone (and in CI) train
    the same small iris model into a temp dir instead."""
    if os.path.exists(LOCAL_MODEL):
        return LOCAL_MODEL
    from sklearn.datasets import load_iris
    from sklearn.ensemble import RandomForestClassifier
    X, y = load_iris(return_X_y=True)
    path = os.path.join(tempfile.mkdtemp(), 'model.pkl')
    joblib.dump(RandomForestClassifier(n_estimators=20, random_state=42).fit(X, y), path)
    return path


MODEL_PATH = _model_path()


@pytest.fixture(autouse=True)
def set_model_env(monkeypatch):
    monkeypatch.setenv('MODEL_PATH', MODEL_PATH)

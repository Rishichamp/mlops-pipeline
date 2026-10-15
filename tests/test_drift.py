import json
import os

import numpy as np
import pandas as pd

from src.monitoring import drift_detector as dd
from src.monitoring.prediction_logger import load_predictions_as_df


def _iris_like(n, shift=0.0, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        'sepal_length': rng.normal(5.8 + shift, 0.8, n),
        'sepal_width':  rng.normal(3.0, 0.4, n),
        'petal_length': rng.normal(3.8 + shift, 1.7, n),
        'petal_width':  rng.normal(1.2, 0.7, n),
    })


def test_small_samples_never_trigger_retraining():
    ref = _iris_like(150)
    # 5 points from the *same* distribution: the old check (n >= 5, mean KS > 0.1)
    # flagged this as drift most of the time
    report = dd.compute_drift_score(ref, _iris_like(5, seed=1))
    assert report['drift_detected'] is False and report['reason'] == 'insufficient_data'


def test_same_distribution_no_drift_and_shift_detected():
    ref = _iris_like(1000, seed=1)
    assert dd.compute_drift_score(ref, _iris_like(500, seed=2))['drift_detected'] is False
    assert dd.compute_drift_score(ref, _iris_like(500, shift=1.5, seed=3))['drift_detected'] is True


def test_no_common_columns_is_valid_json_not_nan():
    report = dd.compute_drift_score(_iris_like(100), pd.DataFrame({'x': range(50)}))
    assert report['drift_detected'] is False
    json.dumps(report, allow_nan=False)


def test_log_reader_skips_bad_lines_and_window(tmp_path):
    path = tmp_path / 'p.jsonl'
    good = {'timestamp': 't', 'features': [5.0, 3.0, 1.0, 0.2], 'prediction': 0, 'confidence': 0.9}
    lines = [json.dumps(good)] * 10 + ['{"truncated": ', json.dumps({**good, 'features': [1, 2]})]
    path.write_text('\n'.join(lines) + '\n')
    assert len(load_predictions_as_df(str(path))) == 10
    assert len(dd.load_production_data(str(path), window=3)) == 3

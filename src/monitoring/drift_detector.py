import pandas as pd
import numpy as np
import os
import json
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FEATURE_COLS = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

def compute_drift_score(reference_df: pd.DataFrame, current_df: pd.DataFrame) -> dict:
    if current_df.empty or len(current_df) < 5:
        logger.warning('Not enough current data for drift detection')
        return {'drift_detected': False, 'drift_score': 0.0, 'reason': 'insufficient_data'}

    from scipy import stats
    drift_scores = {}
    for col in FEATURE_COLS:
        if col not in reference_df.columns or col not in current_df.columns:
            continue
        stat, pvalue = stats.ks_2samp(
            reference_df[col].dropna().values,
            current_df[col].dropna().values
        )
        drift_scores[col] = {
            'statistic': round(float(stat), 4),
            'pvalue':    round(float(pvalue), 4)
        }

    overall_drift  = float(np.mean([v['statistic'] for v in drift_scores.values()]))
    drift_detected = bool(overall_drift > 0.1)   # ← cast to Python bool

    result = {
        'drift_detected': drift_detected,
        'drift_score':    round(overall_drift, 4),
        'feature_drift':  drift_scores,
        'n_reference':    int(len(reference_df)),
        'n_current':      int(len(current_df)),
        'timestamp':      datetime.now().isoformat(),
        'threshold':      0.1,
    }

    logger.info(f'Drift score: {overall_drift:.4f}  detected={drift_detected}')
    return result

def load_reference_data(path: str = 'data/features/features.parquet') -> pd.DataFrame:
    df = pd.read_parquet(path)
    return df[FEATURE_COLS]

def load_production_data(path: str = 'data/predictions/predictions.jsonl') -> pd.DataFrame:
    from src.monitoring.prediction_logger import load_predictions_as_df
    df = load_predictions_as_df(path)
    if df.empty:
        return df
    available = [c for c in FEATURE_COLS if c in df.columns]
    return df[available] if available else pd.DataFrame()

def save_drift_report(report: dict, path: str = 'data/monitoring/drift_report.json'):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(report, f, indent=2)
    logger.info(f'Drift report saved to {path}')

def log_drift_to_mlflow(report: dict, experiment: str = 'mlops-drift-monitoring'):
    import mlflow
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=f'drift_{datetime.now().strftime("%Y%m%d_%H%M")}'):
        mlflow.log_metric('drift_score',         report['drift_score'])
        mlflow.log_metric('drift_detected',      int(report['drift_detected']))
        mlflow.log_metric('n_current_samples',   report.get('n_current', 0))
        for feat, vals in report.get('feature_drift', {}).items():
            mlflow.log_metric(f'drift_{feat}',   vals['statistic'])
        mlflow.log_dict(report, 'drift_report.json')
        logger.info('Drift metrics logged to MLflow')

if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')
    from src.monitoring.prediction_logger import log_prediction

    np.random.seed(99)
    for _ in range(30):
        features = [
            float(np.random.normal(6.0, 0.8)),
            float(np.random.normal(3.0, 0.4)),
            float(np.random.normal(4.5, 1.2)),
            float(np.random.normal(1.5, 0.5)),
        ]
        log_prediction(features, 1, 0.75)

    ref_df  = load_reference_data()
    curr_df = load_production_data()
    report  = compute_drift_score(ref_df, curr_df)
    save_drift_report(report)

    print(f"Drift score:    {report['drift_score']}")
    print(f"Drift detected: {report['drift_detected']}")
    print(f"Feature drift:")
    for feat, vals in report.get('feature_drift', {}).items():
        print(f"  {feat}: KS={vals['statistic']}  p={vals['pvalue']}")

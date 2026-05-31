import mlflow
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix
)
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate_champion_challenger(
    champion_run_id: str,
    challenger_run_id: str,
    X_test, y_test,
    metric: str = 'f1_macro'
) -> dict:
    client = mlflow.tracking.MlflowClient()
    champ_val  = client.get_run(champion_run_id).data.metrics.get(metric, 0)
    chall_val  = client.get_run(challenger_run_id).data.metrics.get(metric, 0)
    winner     = 'challenger' if chall_val > champ_val else 'champion'
    improvement = round((chall_val - champ_val) / max(champ_val, 1e-9) * 100, 2)

    logger.info(f'Champion  {metric}={champ_val:.4f}')
    logger.info(f'Challenger {metric}={chall_val:.4f}')
    logger.info(f'Winner: {winner}  improvement={improvement}%')

    return {
        'champion_score':    champ_val,
        'challenger_score':  chall_val,
        'winner':            winner,
        'improvement_pct':   improvement,
    }

def generate_evaluation_report(model, X_test, y_test, label_names=None) -> dict:
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    report  = {
        'accuracy':         round(accuracy_score(y_test, y_pred), 4),
        'f1_macro':         round(f1_score(y_test, y_pred, average='macro'), 4),
        'f1_weighted':      round(f1_score(y_test, y_pred, average='weighted'), 4),
        'auc_ovr':          round(roc_auc_score(y_test, y_proba, multi_class='ovr'), 4),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'classification_report': classification_report(
            y_test, y_pred,
            target_names=label_names or ['setosa', 'versicolor', 'virginica']
        ),
    }
    return report

if __name__ == '__main__':
    print('Evaluation module ready')
    print('Use generate_evaluation_report(model, X_test, y_test) to evaluate any model')

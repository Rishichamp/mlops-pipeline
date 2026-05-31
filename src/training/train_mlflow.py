import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import joblib
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MLFLOW_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
EXPERIMENT  = os.getenv('MLFLOW_EXPERIMENT_NAME', 'mlops-pipeline')

def setup_mlflow():
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)
    logger.info(f'MLflow tracking: {MLFLOW_URI}  experiment: {EXPERIMENT}')

def load_features(path: str = 'data/features/features.parquet') -> tuple:
    df = pd.read_parquet(path)
    feature_cols = [c for c in df.columns if c not in ['target', 'species']]
    X = df[feature_cols].values
    y = df['target'].values
    logger.info(f'Loaded features: X={X.shape}  y={y.shape}')
    return X, y, feature_cols

def evaluate_model(model, X_test, y_test) -> dict:
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    return {
        'accuracy': round(float(accuracy_score(y_test, y_pred)), 4),
        'f1_macro': round(float(f1_score(y_test, y_pred, average='macro')), 4),
        'auc_ovr':  round(float(roc_auc_score(y_test, y_proba, multi_class='ovr')), 4),
    }

def train_single_run(params: dict, X_train, X_test, y_train, y_test, feature_cols, run_name: str = None) -> str:
    with mlflow.start_run(run_name=run_name) as run:
        model_type = params.pop('model_type', 'random_forest')
        mlflow.log_params({'model_type': model_type, **params})

        if model_type == 'random_forest':
            model = RandomForestClassifier(**params, random_state=42, n_jobs=1)
        elif model_type == 'gradient_boosting':
            model = GradientBoostingClassifier(**params, random_state=42)
        else:
            model = LogisticRegression(**params, random_state=42, max_iter=1000)

        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_macro')
        mlflow.log_metric('cv_f1_mean', round(float(cv_scores.mean()), 4))
        mlflow.log_metric('cv_f1_std',  round(float(cv_scores.std()), 4))

        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        mlflow.log_metrics(metrics)

        mlflow.log_param('feature_count', len(feature_cols))
        mlflow.log_param('train_size', len(X_train))
        mlflow.set_tag('feature_cols', ','.join(feature_cols))

        # v2-compatible: log model without input_example to avoid flask dependency
        mlflow.sklearn.log_model(
            model,
            artifact_path='model',
            registered_model_name='iris-classifier',
        )

        logger.info(f'Run {run.info.run_id[:8]}  {model_type}  metrics={metrics}')
        return run.info.run_id

def run_hyperparameter_search(X_train, X_test, y_train, y_test, feature_cols) -> list:
    trials = [
        {'model_type': 'random_forest',    'n_estimators': 50,  'max_depth': 3},
        {'model_type': 'random_forest',    'n_estimators': 100, 'max_depth': 5},
        {'model_type': 'random_forest',    'n_estimators': 200, 'max_depth': 7},
        {'model_type': 'random_forest',    'n_estimators': 100, 'max_depth': None},
        {'model_type': 'gradient_boosting','n_estimators': 50,  'max_depth': 3, 'learning_rate': 0.1},
        {'model_type': 'gradient_boosting','n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.05},
        {'model_type': 'gradient_boosting','n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.1},
        {'model_type': 'logistic',         'C': 0.1},
        {'model_type': 'logistic',         'C': 1.0},
        {'model_type': 'logistic',         'C': 10.0},
    ]
    run_ids = []
    for i, params in enumerate(trials):
        run_name = f'trial_{i+1:02d}_{params["model_type"]}'
        run_id   = train_single_run(dict(params), X_train, X_test, y_train, y_test, feature_cols, run_name)
        run_ids.append(run_id)
    return run_ids

def get_best_run(run_ids: list, metric: str = 'f1_macro') -> tuple:
    client = mlflow.tracking.MlflowClient()
    best_run_id, best_val = None, -1
    for rid in run_ids:
        val = client.get_run(rid).data.metrics.get(metric, -1)
        if val > best_val:
            best_val, best_run_id = val, rid
    logger.info(f'Best run: {best_run_id[:8]}  {metric}={best_val}')
    return best_run_id, best_val

def register_best_model(run_id: str, model_name: str = 'iris-classifier') -> str:
    client    = mlflow.tracking.MlflowClient()
    model_uri = f'runs:/{run_id}/model'
    mv = mlflow.register_model(model_uri, model_name)
    client.set_registered_model_tag(model_name, 'pipeline', 'mlops-phase3')
    logger.info(f'Registered version {mv.version}  status={mv.status}')
    return mv.version

def promote_to_production(model_name: str, version: str, accuracy_threshold: float = 0.85) -> bool:
    client = mlflow.tracking.MlflowClient()
    mv     = client.get_model_version(model_name, version)
    run    = client.get_run(mv.run_id)
    acc    = run.data.metrics.get('accuracy', 0)
    if acc < accuracy_threshold:
        logger.warning(f'Model v{version} accuracy {acc:.4f} below threshold. NOT promoted.')
        return False
    client.set_model_version_tag(model_name, version, 'stage', 'Production')
    logger.info(f'Model {model_name} v{version} promoted to Production (accuracy={acc:.4f})')
    return True

if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')
    from src.ingestion.data_loader  import load_raw_data
    from src.ingestion.validator    import validate_raw_data
    from src.features.feature_engineering import engineer_features, save_features

    df       = load_raw_data()
    df       = validate_raw_data(df)
    features = engineer_features(df)
    save_features(features)

    setup_mlflow()
    X, y, feature_cols = load_features()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    run_ids = run_hyperparameter_search(X_train, X_test, y_train, y_test, feature_cols)
    best_run_id, best_f1 = get_best_run(run_ids)
    version  = register_best_model(best_run_id)
    promoted = promote_to_production('iris-classifier', version)

    print(f'Best run: {best_run_id}')
    print(f'Best F1:  {best_f1}')
    print(f'Version:  {version}')
    print(f'Promoted: {promoted}')

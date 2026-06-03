from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.utils.dates import days_ago
from datetime import timedelta
import sys, os

sys.path.insert(0, '/opt/airflow')

default_args = {
    'owner': 'mlops',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'email_on_failure': False,
}

def task_generate_predictions(**context):
    import numpy as np
    from src.monitoring.prediction_logger import log_prediction
    np.random.seed(int(datetime.now().timestamp()) % 1000)
    n = 50
    for _ in range(n):
        features = [
            float(np.random.normal(5.8, 0.9)),
            float(np.random.normal(3.0, 0.45)),
            float(np.random.normal(4.2, 1.3)),
            float(np.random.normal(1.3, 0.6)),
        ]
        log_prediction(features, 1, 0.75)
    context['ti'].xcom_push(key='n_predictions', value=n)
    print(f'Generated {n} synthetic production predictions')

def task_upload_predictions(**context):
    from src.monitoring.prediction_logger import upload_predictions_to_minio
    upload_predictions_to_minio()
    print('Predictions uploaded to MinIO')

def task_compute_drift(**context):
    os.environ['MLFLOW_TRACKING_URI']    = 'http://mlflow:5000'
    os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://minio:9000'
    os.environ['AWS_ACCESS_KEY_ID']      = 'minioadmin'
    os.environ['AWS_SECRET_ACCESS_KEY']  = 'minioadmin'
    os.environ['AWS_DEFAULT_REGION']     = 'us-east-1'

    from src.monitoring.drift_detector import (
        load_reference_data, load_production_data,
        compute_drift_score, save_drift_report, log_drift_to_mlflow
    )

    ref_df   = load_reference_data('/opt/airflow/data/features/features.parquet')
    curr_df  = load_production_data('/opt/airflow/data/predictions/predictions.jsonl')
    report   = compute_drift_score(ref_df, curr_df)

    save_drift_report(report, '/opt/airflow/data/monitoring/drift_report.json')
    log_drift_to_mlflow(report)

    context['ti'].xcom_push(key='drift_score',    value=report['drift_score'])
    context['ti'].xcom_push(key='drift_detected', value=report['drift_detected'])
    print(f"Drift score={report['drift_score']}  detected={report['drift_detected']}")

def task_check_drift(**context):
    drift_detected = context['ti'].xcom_pull(key='drift_detected', task_ids='compute_drift')
    if drift_detected:
        print('DRIFT DETECTED — triggering retraining')
        return 'retrain_model'
    print('No drift detected — model healthy')
    return 'no_action'

def task_no_action(**context):
    drift_score = context['ti'].xcom_pull(key='drift_score', task_ids='compute_drift')
    print(f'Model is healthy. Drift score={drift_score:.4f} below threshold 0.1')

def task_retrain(**context):
    os.environ['MLFLOW_TRACKING_URI']    = 'http://mlflow:5000'
    os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://minio:9000'
    os.environ['AWS_ACCESS_KEY_ID']      = 'minioadmin'
    os.environ['AWS_SECRET_ACCESS_KEY']  = 'minioadmin'
    os.environ['AWS_DEFAULT_REGION']     = 'us-east-1'
    os.environ['GIT_PYTHON_REFRESH']     = 'quiet'

    from src.training.train_mlflow import (
        setup_mlflow, load_features, run_hyperparameter_search,
        get_best_run, register_best_model, promote_to_production
    )
    from sklearn.model_selection import train_test_split

    setup_mlflow()
    X, y, feature_cols = load_features('/opt/airflow/data/features/features.parquet')
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    run_ids          = run_hyperparameter_search(X_train, X_test, y_train, y_test, feature_cols)
    best_run_id, f1  = get_best_run(run_ids)
    version          = register_best_model(best_run_id)
    promoted         = promote_to_production('iris-classifier', version)

    context['ti'].xcom_push(key='new_version', value=version)
    print(f'Retrained. New version={version}  F1={f1:.4f}  promoted={promoted}')

from datetime import datetime

with DAG(
    dag_id='drift_detection_dag',
    default_args=default_args,
    description='Daily drift detection and auto-retraining',
    schedule_interval='@daily',
    start_date=days_ago(1),
    catchup=False,
    tags=['mlops', 'drift', 'monitoring'],
) as dag:

    gen_preds  = PythonOperator(task_id='generate_predictions', python_callable=task_generate_predictions)
    upload     = PythonOperator(task_id='upload_predictions',   python_callable=task_upload_predictions)
    drift      = PythonOperator(task_id='compute_drift',        python_callable=task_compute_drift)
    gate       = BranchPythonOperator(task_id='check_drift',    python_callable=task_check_drift)
    no_action  = PythonOperator(task_id='no_action',            python_callable=task_no_action)
    retrain    = PythonOperator(task_id='retrain_model',        python_callable=task_retrain)

    gen_preds >> upload >> drift >> gate >> [no_action, retrain]

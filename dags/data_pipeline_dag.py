from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from datetime import timedelta
import sys, os

sys.path.insert(0, '/opt/airflow')

default_args = {
    'owner': 'mlops',
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
    'email_on_failure': False,
}

def task_ingest(**context):
    from src.ingestion.data_loader import load_raw_data
    df = load_raw_data(output_path='/opt/airflow/data/raw/iris.csv')
    context['ti'].xcom_push(key='row_count', value=len(df))
    print(f'Ingested {len(df)} rows')

def task_validate(**context):
    from src.ingestion.validator import validate_from_csv
    df = validate_from_csv('/opt/airflow/data/raw/iris.csv')
    print(f'Validation passed for {len(df)} rows')

def task_engineer(**context):
    from src.ingestion.validator import validate_from_csv
    from src.features.feature_engineering import engineer_features, save_features
    df = validate_from_csv('/opt/airflow/data/raw/iris.csv')
    features = engineer_features(df, scaler_path='/opt/airflow/artifacts/scaler.pkl', fit_scaler=True)
    save_features(features, output_path='/opt/airflow/data/features/features.parquet')
    print(f'Features engineered: {features.shape}')

def task_upload(**context):
    from src.ingestion.minio_client import upload_file, ensure_bucket
    from datetime import datetime
    ensure_bucket('mlops-features')
    date_str = datetime.now().strftime('%Y%m%d')
    upload_file(
        local_path='/opt/airflow/data/features/features.parquet',
        bucket='mlops-features',
        key=f'iris/features_{date_str}.parquet'
    )
    print('Features uploaded to MinIO')

def task_train(**context):
    # Set ALL required env vars BEFORE importing mlflow
    os.environ['MLFLOW_TRACKING_URI']    = 'http://mlflow:5000'
    os.environ['MLFLOW_EXPERIMENT_NAME'] = 'mlops-pipeline'
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
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    run_ids          = run_hyperparameter_search(X_train, X_test, y_train, y_test, feature_cols)
    best_run_id, best_f1 = get_best_run(run_ids)
    version          = register_best_model(best_run_id)
    promoted         = promote_to_production('iris-classifier', version)

    context['ti'].xcom_push(key='best_run_id',    value=best_run_id)
    context['ti'].xcom_push(key='best_f1',        value=best_f1)
    context['ti'].xcom_push(key='model_version',  value=version)
    context['ti'].xcom_push(key='promoted',       value=promoted)
    print(f'Training complete. Best F1={best_f1:.4f}  version={version}  promoted={promoted}')

def task_report(**context):
    best_run_id = context['ti'].xcom_pull(key='best_run_id',   task_ids='train')
    best_f1     = context['ti'].xcom_pull(key='best_f1',       task_ids='train')
    version     = context['ti'].xcom_pull(key='model_version', task_ids='train')
    promoted    = context['ti'].xcom_pull(key='promoted',      task_ids='train')
    row_count   = context['ti'].xcom_pull(key='row_count',     task_ids='ingest')
    print('=' * 50)
    print('PIPELINE REPORT')
    print('=' * 50)
    print(f'Training samples : {row_count}')
    print(f'Best run ID      : {best_run_id}')
    print(f'Best F1 (macro)  : {best_f1:.4f}')
    print(f'Model version    : {version}')
    print(f'Promoted         : {promoted}')
    print(f'MLflow UI        : http://localhost:5000')
    print('=' * 50)

with DAG(
    dag_id='mlops_data_pipeline',
    default_args=default_args,
    description='Full MLOps pipeline: ingest, validate, features, train, register',
    schedule_interval='@daily',
    start_date=days_ago(1),
    catchup=False,
    tags=['mlops', 'training', 'mlflow'],
) as dag:

    ingest   = PythonOperator(task_id='ingest',            python_callable=task_ingest)
    validate = PythonOperator(task_id='validate',          python_callable=task_validate)
    engineer = PythonOperator(task_id='engineer_features', python_callable=task_engineer)
    upload   = PythonOperator(task_id='upload_to_minio',   python_callable=task_upload)
    train    = PythonOperator(task_id='train',             python_callable=task_train)
    report   = PythonOperator(task_id='report',            python_callable=task_report)

    ingest >> validate >> engineer >> upload >> train >> report
